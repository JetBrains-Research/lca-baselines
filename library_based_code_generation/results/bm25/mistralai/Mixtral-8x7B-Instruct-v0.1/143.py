 from py_shiny import App, Inputs, r\_a, funcA, call\_a, first, column, second, Input, SelectInput, textInput, tabset, tabPanel, expect\_column\_text, layout\_column\_wrap, expect\_column\_labels, in\_selected, expect\_locator\_values\_in\_list, set, _set, expect\_locator\_contains\_values\_in\_list, debug

app\_ui = App(debug=True)

with app\_ui:
column(
width=3,
Inputs(
id="input1",
label="Input 1",
type="text",
value="Initial value 1"
),
Inputs(
id="input2",
label="Input 2",
type="text",
value="Initial value 2"
)
)

column(
width=3,
Inputs(
id="input3",
label="Input 3",
value="Dependent value 3",
disabled=True
),
Inputs(
id="input4",
label="Input 4",
value="Dependent value 4",
disabled=True
)
)

column(
width=6,
tabset(
id="tabset1",
tabPanel(
"Tab 1",
Inputs(
id="input5",
label="Input 5",
value="Dependent value 5",
disabled=True
)
),
tabPanel(
"Tab 2",
Inputs(
id="input6",
label="Input 6",
value="Dependent value 6",
disabled=True
)
)
)
)

def server\_logic(input, output, session):
@should\_continue(session)
def update\_inputs():
input1 = input["input1"]()
input2 = input["input2"]()

if first(input1) and first(input2):
# Update second column inputs
set(session, "input3", input1)
set(session, "input4", input2)

# Update third column inputs
tab\_name = in\_selected(session, "tabset1")
if tab\_name == "Tab 1":
set(session, "input5", input1 + " " + input2)
elif tab\_name == "Tab 2":
set(session, "input6", input2 + " " + input1)

return True

@_manage\_inputs(session)
def manage\_inputs():
if not _should\_suspend(session):
update\_inputs()

if __name__ == "__main__":
app\_ui.run()