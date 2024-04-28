use either::*;
use ndarray::{Array2, ArrayView2};
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
    const TABLE_FIRST_ROWS: usize = 4;
    const IMAGE_SIZE: (u32, u32) = (300, 300);
    const HEATMAPS_PER_ROW: u8 = 7;

    // preview first rows of data source
    pub fn table_preview(data: Either<Option<&Box<Array2<f64>>>, (&Array2<f64>, &Array2<f64>)>, 
                headers: Option<&Vec<String>>) {

        let data_rows = match data {
            Left(l) => l.map_or(0, |a| a.nrows()),
            Right(r) => r.0.nrows(),
        };

        // print first rows, whichever is shorter
        let n_rows = std::cmp::min(data_rows, Self::TABLE_FIRST_ROWS);

        if n_rows == 0 { return } // exit early

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

        let mut table_row: Vec<&f64>;
        let (mut x_row, mut y_row, mut view);

        // construct table based on the two different data input types
        for i in 0..n_rows {
            match data {
                Left(l) => {
                    view = l.as_ref().unwrap().row(i);
                    table_row = view.iter().collect();
                },
                Right(r) => {
                    x_row = r.0.row(i);
                    table_row = x_row.iter().collect();
                    y_row = r.1.row(i);
                    table_row.extend(y_row.iter());
                }
            }

           table.add_row(table_row);
        }

        println!("{table}");
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
