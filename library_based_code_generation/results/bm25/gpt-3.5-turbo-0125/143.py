```python
import py_shiny as ps

app = ps.App(debug=True)

with app:
    ps.column(
        ps.inputs(ps._manage_inputs(ps.should_continue(ps.r_a(funcA(call_a(first=2)))))),
        ps.column(
            ps._find_first_option(ps.expect_column_text(layout_column_wrap(ps.expect_column_labels(in_selected(ps.expect_locator_values_in_list(set(ps._set()))))))),
            ps._find_first_option(ps.expect_column_text(layout_column_wrap(ps.expect_column_labels(in_selected(ps.expect_locator_values_in_list(set(ps._set())))))))
        ),
        ps.column(
            ps._find_first_option(ps.expect_column_text(layout_column_wrap(ps.expect_column_labels(in_selected(ps.expect_locator_values_in_list(set(ps._set()))))))),
            ps.tabset()
        )
    )

app.run()
```