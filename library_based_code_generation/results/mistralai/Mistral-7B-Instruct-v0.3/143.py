 Here is a simple example of a web application using py-shiny that meets your requirements. Please note that you need to have py-shiny installed before running this code. You can install it using `pip install py-shiny`.

```python
import shiny

ui = fluidPage(
    fluidRow(
        column(3,
            shiny.ui.inputText("input1", "Input 1"),
            shiny.ui.inputText("input2", "Input 2")
        ),
        column(6,
            shiny.ui.fluidPage(
                for i in range(2):
                    shiny.ui.column(6,
                        shiny.ui.inputText("input_{}".format(i+3), "Input {}".format(i+3))
                ),
                shiny.ui.tabsetPanel(
                    for i in range(3):
                        shiny.ui.tabPanel("Tab {}".format(i+1),
                            for j in range(2):
                                shiny.ui.inputText("input_{}_{}".format(i+4, j+7), "Input {}-{}".format(i+4, j+7))
                        )
                )
            )
        )
    )
)

server = shiny.ServerFunction(
    function(input, output, session):
        for i in range(2):
            observeEvent(input$input1, {
                updateInputValues(session, "input_{}".format(i+3), input$input1)
            })
            observeEvent(input$input2, {
                updateInputValues(session, "input_{}".format(i+3), input$input2)
            })

        for i in range(3):
            for j in range(2):
                observeEvent(input$input_{i+4}, {
                    updateInputValues(session, "input_{}_{}".format(i+4, j+7), input$input_{i+4})
                })

        def updateInputValues(session, input_id, value):
            session.sendCustomMessage(type="update", input_id=input_id, value=value)

)

app_run = shiny.App(ui=ui, server=server)
app_run.run(debug=True)
```

This code creates a web application with three columns. The first column contains two input text boxes (`input1` and `input2`). The second column contains six input text boxes (`input_3` to `input_8`), which are controlled by `input1` and `input2`. The third column contains a tabset with three tabs, each containing two input text boxes (`input_9` to `input_24`). The server function updates the inputs in the second and third columns based on the values of `input1` and `input2`. The application is run in debug mode.