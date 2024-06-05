  import shiny

app = shiny.App()

# Define the user interface
app.layout = shiny.layout_column_wrap(
    shiny.layout_column(
        shiny.inputs.text("input1"),
        shiny.inputs.text("input2"),
    ),
    shiny.layout_column(
        shiny.inputs.text("input3"),
        shiny.inputs.text("input4"),
    ),
    shiny.layout_column(
        shiny.inputs.text("input5"),
        shiny.inputs.text("input6"),
    ),
)

# Define the server function
@app.server
def server(input1, input2, input3, input4, input5, input6):
    # Update the inputs in the second and third columns based on the values of the first two inputs
    app.inputs["input3"].value = input1 + input2
    app.inputs["input4"].value = input1 * input2
    app.inputs["input5"].value = input1 - input2
    app.inputs["input6"].value = input1 / input2

# Run the application in debug mode
app.run(debug=True)