use ndarray::ArrayView2;
use comfy_table::{Table, ContentArrangement};
use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::UTF8_FULL;
use plotters::prelude::*;
use plotters::backend::BitMapBackend as BMB;
use colorous::{PLASMA, GREYS, TURBO};
use viuer::{print_from_file, Config};

use crate::algebra::AlgebraExt;

pub struct Visualize;

impl Visualize {
    const ASCII_ART_SIZE: usize = 15;
    const TABLE_FIRST_ROWS: usize = 4;
    const IMAGE_SIZE: (u32, u32) = (300, 300);
    const HEATMAPS_PER_ROW: u8 = 7;

    // preview first rows of data source
    pub fn table_preview(data: &ArrayView2<f64>, 
                         headers: Option<&Vec<String>>, ascii_art: bool) {

        // print first rows, whichever is shorter
        let max = if ascii_art { Self::ASCII_ART_SIZE } else { Self::TABLE_FIRST_ROWS };
        let n_rows = std::cmp::min(data.nrows(), max);
        if n_rows == 0 { return } // exit early, nothing to preview

        let mut table = Table::new();

        table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_width(120);

        // set table header
        if headers.is_some() {
            table.set_header(*headers.as_ref().unwrap());
        }

        let mut row_view;

        // construct table based on the two different data input types
        for i in 0..n_rows {
            row_view = data.row(i);

            if ascii_art {
                // if row's sum is 0 don't add it -- to compact table
                if row_view.sum() != 0. {
                    table.add_row(
                        row_view.into_iter()
                            .map(|v| Self::float_to_ascii(&v))
                            .collect::<Vec<char>>()
                    );
                }
            } else {
                table.add_row(row_view.iter().collect::<Vec<&f64>>());
            }
        }

        println!("{table}");
    }

    fn float_to_ascii(val: &f64) -> char {
        match (*val * 100.0) as u8 {
            0 => ' ',
            100 => '#',
            v if v > 0 && v <= 50 => '%',
            v if v > 50 && v < 100 => '$',
            _ => ' ',
        }
    }

    pub fn heatmap_row(image: &ArrayView2<f64>, mut index: u8) {
        index = index % Self::HEATMAPS_PER_ROW;  // only accept < 10 heatmap images per row
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
            let color = color_scale.eval_continuous(data_value_scaled as f64);
            empty_cell.fill(&RGBColor(color.r, color.g, color.b)).unwrap();
        };

        draw_area.present().unwrap();

        let col_offset = index * 25u8;

        let conf = Config {
            x: col_offset as u16, // terminal col offset
            y: 5,  // terminal row offset
            width: Some(20), // term cell dimension width
            height: Some(10), // term cell dimension height
            ..Default::default()
        };

        print_from_file(&filename, &conf).expect("Unable to print image");
    }
}
