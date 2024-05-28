use ndarray::{Array2, ArrayView2};
use comfy_table::{Table, ContentArrangement};
use comfy_table::{modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL};
use plotters::{prelude::*, backend::BitMapBackend as BMB};
use colorous::{PLASMA, GREYS, TURBO};
use viuer::{print_from_file, Config};

use crate::algebra::AlgebraExt;

pub trait Peek {
    fn peek(x: &Array2<f64>, txt: Option<&str>);
}

pub struct Visualize;

impl Visualize {
    const ASCII_ART_SIZE: usize = 15;
    const TABLE_FIRST_ROWS: usize = 4;
    const IMAGE_SIZE: (u32, u32) = (300, 300);
    const HEATMAPS_PER_ROW: u8 = 7;

    // preview first rows of data source
    pub fn table_preview(data: &ArrayView2<f64>, headers: Option<&Vec<String>>,
                         ascii_art: bool, text: Option<&str>) -> Table {
        use ndarray_stats::QuantileExt;

        text.inspect(|t| println!("{t}"));

        // print first rows, whichever is shorter
        let max = if ascii_art { Self::ASCII_ART_SIZE } else { Self::TABLE_FIRST_ROWS };
        let n_rows = std::cmp::min(data.nrows(), max);

        let mut table = Table::new();

        if n_rows == 0 { return table } // exit early, nothing to preview

        table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_width(120);

        // set table header
        if let Some(h) = &headers {
            table.set_header(*h);
        }

        let mut row_view;
        let mut cur; // track if current row or previous row is all zeros
        let mut prev = false;

        // construct table based on the two different data input types
        for i in 0..n_rows {
            row_view = data.row(i);

            if ascii_art {
                cur = row_view.sum() == 0. &&
                    *QuantileExt::max(&row_view).unwrap() == 0.;

                // if current row with nonzero terms, add it OR 
                // if previous row wasn't all zeros, add it -- to slightly compact table
                if !cur || !prev {
                    table.add_row(
                        row_view.into_iter()
                            .map(Self::float_to_ascii)
                            .collect::<Vec<char>>()
                    );
                }

                prev = cur;
            } else {
                table.add_row(row_view.iter().collect::<Vec<&f64>>());
            }
        }

        println!("{}", &table);
        table
    }

    fn float_to_ascii(val: &f64) -> char {
        match (*val * 100.0) as u8 {
            0 => ' ',
            100 => '#',
            v if v > 0 && v <= 50 => '%',
            v if v > 50 && v <= 75 => '@',
            v if v > 75 && v < 100 => 'X',
            _ => ' ',
        }
    }

    pub fn heatmap_row(image: &ArrayView2<f64>, index: u8) {
        let col_index = index % Self::HEATMAPS_PER_ROW;  // only accept < 10 heatmap images per row
        let row_index = index / Self::HEATMAPS_PER_ROW;
        let (n_rows, n_cols) = (image.shape()[0], image.shape()[1]);

        let filename = format!("/tmp/tmp-heatmap{}.png", &index);
        let draw_area = BMB::new(&filename, Self::IMAGE_SIZE).into_drawing_area();
        let empty_cells = draw_area.split_evenly((n_cols, n_rows));

        // Scaling values
        let normalized_image = image.map(|v| v/(n_cols as f64));
        let max_value: f64 = normalized_image.max();

        // Add continuous color scale to assist with mapping data value
        let color_scale = match index % 3 { 0 => PLASMA, 1 => GREYS, _ => TURBO };

        // map data to color, fill cell with color given color's rgb value
        for (empty_cell, data_value) in empty_cells.iter().zip(normalized_image.iter()) {
            let data_value_scaled = data_value.sqrt() / max_value.sqrt();
            let color = color_scale.eval_continuous(data_value_scaled);
            empty_cell.fill(&RGBColor(color.r, color.g, color.b)).unwrap();
        };

        // save heatmap image to file
        draw_area.present().unwrap();

        let col_offset = col_index * 25u8;
        let row_offset = row_index * 10u8;

        // scale and offset image and reload from file
        let conf = Config {
            x: col_offset as u16, // terminal col offset
            y: 5 + row_offset as i16,  // terminal row offset
            width: Some(20), // term cell dimension width
            height: Some(10), // term cell dimension height
            ..Default::default()
        };

        print_from_file(&filename, &conf).expect("Unable to print image");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    pub fn regular_table_pre() {
        let input = arr2(&[[9.0, 5.3, 1.4, 7.2]]);
        let headers = vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()];

        let t = Visualize::table_preview(
            &input.view(), Some(&headers),
            false, None);

        let out =
"╭───┬─────┬─────┬─────╮
│ A ┆ B   ┆ C   ┆ D   │
╞═══╪═════╪═════╪═════╡
│ 9 ┆ 5.3 ┆ 1.4 ┆ 7.2 │
╰───┴─────┴─────┴─────╯";

        assert_eq!(t.to_string(), out.to_string());
    }

    #[test]
    pub fn ascii_table_pre() {
        let input = arr2(&[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.4, 0.7, 0.85, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.81, 0.82, 0.8, 0.8, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.8, 0.6, 0.6, 0.8, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.8, 0.6, 0.8, 0.6, 0.0, 0.0],
            [0.8, 0.8, 0.8, 0.8, 0.8, 0.6, 0.8, 0.8, 0.8],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]);

        // view in regular table view (only prints first four rows)
        let t = Visualize::table_preview(
            &input.view(), None,
            false, None);

        let out1 =
"╭───┬───┬─────┬──────┬──────┬──────┬─────┬───┬───╮
│ 0 ┆ 0 ┆ 0   ┆ 0    ┆ 0    ┆ 0    ┆ 0   ┆ 0 ┆ 0 │
├╌╌╌┼╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌┼╌╌╌┤
│ 0 ┆ 0 ┆ 0   ┆ 0    ┆ 0    ┆ 0    ┆ 0   ┆ 0 ┆ 0 │
├╌╌╌┼╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌┼╌╌╌┤
│ 0 ┆ 0 ┆ 0   ┆ 0.4  ┆ 0.7  ┆ 0.85 ┆ 0   ┆ 0 ┆ 0 │
├╌╌╌┼╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌┼╌╌╌┤
│ 0 ┆ 0 ┆ 0.8 ┆ 0.81 ┆ 0.82 ┆ 0.8  ┆ 0.8 ┆ 0 ┆ 0 │
╰───┴───┴─────┴──────┴──────┴──────┴─────┴───┴───╯";

        assert_eq!(t.to_string(), out1.to_string());

        // view same data but in ascii table view
        let t = Visualize::table_preview(
            &input.view(), None,
            true, None);

        // image of a hat
        let out2 =
            "╭───┬───┬───┬───┬───┬───┬───┬───┬───╮
│   ┆   ┆   ┆   ┆   ┆   ┆   ┆   ┆   │
├╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┤
│   ┆   ┆   ┆ % ┆ @ ┆ X ┆   ┆   ┆   │
├╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┤
│   ┆   ┆ X ┆ X ┆ X ┆ X ┆ X ┆   ┆   │
├╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┤
│   ┆   ┆ X ┆ X ┆ @ ┆ @ ┆ X ┆   ┆   │
├╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┤
│   ┆   ┆ X ┆ X ┆ @ ┆ X ┆ @ ┆   ┆   │
├╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┤
│ X ┆ X ┆ X ┆ X ┆ X ┆ @ ┆ X ┆ X ┆ X │
├╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┼╌╌╌┤
│   ┆   ┆   ┆   ┆   ┆   ┆   ┆   ┆   │
╰───┴───┴───┴───┴───┴───┴───┴───┴───╯";

            assert_eq!(t.to_string(), out2.to_string());
    }
}
