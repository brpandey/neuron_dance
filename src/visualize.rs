use either::*;

use ndarray::Array2;

use comfy_table::{Table, ContentArrangement};
use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::UTF8_FULL;

pub struct Visualize;

impl Visualize {
    const FIRST_ROWS: usize = 4;

    // preview first rows of data source
    pub fn preview(data: Either<Option<&Box<Array2<f64>>>, (&Array2<f64>, &Array2<f64>)>, 
                headers: Option<&Vec<String>>) {

        let data_rows = match data {
            Left(l) => l.map_or(0, |a| a.nrows()),
            Right(r) => r.0.nrows(),
        };

        // print first rows, whichever is shorter
        let n_rows = std::cmp::min(data_rows, Self::FIRST_ROWS);

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

        if n_rows > 0 {
            println!("{table}");
        }
    }
}
