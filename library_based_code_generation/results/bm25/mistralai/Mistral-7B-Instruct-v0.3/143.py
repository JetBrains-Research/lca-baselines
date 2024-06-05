 Here is a basic example of a web application using py-shiny that follows your instructions. Please note that you may need to install the necessary packages and adjust the code according to your specific needs.

```python
import shiny

ui = fluidPage(
    column(width=3,
           shiny.InputText("input1", value="Initial Value 1"),
           shiny.InputText("input2", value="Initial Value 2")
    ),
    column(width=6,
           shiny.ui.FluidPage(
               shiny.ui.Column(
                   for i in range(1, 10):
                       shiny.InputText("input_{}".format(i), value="Initial Value {}".format(i))
               )
           ).server,
           shiny.TabsetPanel(
               shiny.TabPanel("Tab 1",
                              shiny.ui.Column(
                                  for i in range(1, 5):
                                      shiny.InputText("input_{}".format(i + 10), value="Initial Value {}".format(i + 10))
                              )
               ),
               shiny.TabPanel("Tab 2",
                              shiny.ui.Column(
                                  for i in range(1, 5):
                                      shiny.InputText("input_{}".format(i + 15), value="Initial Value {}".format(i + 15))
                              )
               )
           )
    )
)

def server(input, output, session):
    def update_inputs(input1, input2):
        for i in range(1, 10):
            session.set_input("input_{}".format(i), input1 + i)
        for i in range(1, 5):
            session.set_input("input_{}".format(i + 10), input2 + i)

        for i in range(1, 5):
            session.set_input("input_{}".format(i + 15), input1 * i)

    shiny.reactive_values({"input1": "Initial Value 1", "input2": "Initial Value 2"})
    shiny.observe({
        shiny.debounce(1000, update_inputs, input$input1, input$input2)
    })

shiny.run_server(debug=True, ui=ui, server=server)
```

This code creates a web application with three columns as specified. The first two inputs in the first column control the other inputs on the page. The second column contains a set of inputs that are controlled by the first two inputs. The third column contains a tabset with two tabs, each containing a set of inputs. The server function updates the inputs in the second and third columns based on the values of the first two inputs. The application is run in debug mode.