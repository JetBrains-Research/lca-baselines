import py_shiny as ps

app = ps.app()

ui = ps.ui(
    ps.column(
        ps.inputs(),
        ps.inputs()
    ),
    ps.column(
        ps.inputs(),
        ps.inputs()
    ),
    ps.column(
        ps.inputs(),
        ps.tabset()
    )
)

def server(input, output):
    # Update inputs in the second and third columns based on the values of the first two inputs
    pass

app.run(ui, server, debug=True)