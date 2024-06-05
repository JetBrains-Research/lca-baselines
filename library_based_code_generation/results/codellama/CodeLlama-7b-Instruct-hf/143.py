  import shiny

app = shiny.App(
    ui = shiny.UI(
        shiny.Column(
            shiny.Row(
                shiny.Input(shiny.TextInput, "Input 1"),
                shiny.Input(shiny.TextInput, "Input 2")
            ),
            shiny.Column(
                shiny.Row(
                    shiny.Input(shiny.TextInput, "Input 3"),
                    shiny.Input(shiny.TextInput, "Input 4")
                )
            ),
            shiny.Column(
                shiny.Row(
                    shiny.Input(shiny.TextInput, "Input 5"),
                    shiny.Input(shiny.TextInput, "Input 6")
                ),
                shiny.Tabset(
                    shiny.Tab("Tab 1", shiny.Column(
                        shiny.Row(
                            shiny.Input(shiny.TextInput, "Input 7"),
                            shiny.Input(shiny.TextInput, "Input 8")
                        )
                    )),
                    shiny.Tab("Tab 2", shiny.Column(
                        shiny.Row(
                            shiny.Input(shiny.TextInput, "Input 9"),
                            shiny.Input(shiny.TextInput, "Input 10")
                        )
                    ))
                )
            )
        )
    ),
    server = shiny.Server(
        shiny.Function(
            shiny.Input(shiny.TextInput, "Input 1"),
            shiny.Input(shiny.TextInput, "Input 2"),
            shiny.Function(
                shiny.Input(shiny.TextInput, "Input 3"),
                shiny.Input(shiny.TextInput, "Input 4"),
                shiny.Function(
                    shiny.Input(shiny.TextInput, "Input 5"),
                    shiny.Input(shiny.TextInput, "Input 6"),
                    shiny.Function(
                        shiny.Input(shiny.TextInput, "Input 7"),
                        shiny.Input(shiny.TextInput, "Input 8"),
                        shiny.Function(
                            shiny.Input(shiny.TextInput, "Input 9"),
                            shiny.Input(shiny.TextInput, "Input 10")
                        )
                    )
                )
            )
        )
    )
)

if __name__ == "__main__":
    app.run(debug=True)