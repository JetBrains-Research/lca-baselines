 Here is the Python code using the manim library as per your instructions:

```python
from manim import *

class Scene1(Scene):
    def construct(self):
        title = MathTex("Title").scale(2)
        equation = MathTex("f(x) = x^2").next_to(title, DOWN, buff=1)
        self.play(Write(title), Write(equation))
        self.play(Transform(title, title.copy().scale(1.5)), FadeOut(equation))

        grid = Axes(animate=True, x_range=[-5, 5, 1], y_range=[-5, 5, 1])
        grid_title = MathTex("Grid").next_to(grid, UP, buff=1)
        non_linear_function = function_plot(lambda x: x**2, x_range=[-5, 5], y_range=[-50, 50])
        non_linear_function.shift(grid.get_y_axis_of().get_end())
        self.play(Write(grid_title), Write(non_linear_function))
        self.play(ApplyMethod(non_linear_function.shift, grid.get_x_axis_of().get_end(), run_time=3))
        self.play(Transform(grid_title, grid_title.copy().next_to(non_linear_function, RIGHT, buff=1)))

class Scene2(Scene):
    def construct(self):
        square = Square(side_length=2)
        self.play(Create(square))
        self.play(Transform(square, Circle(radius=2)))
        self.play(FadeOut(square))

class Scene3(Scene):
    def construct(self):
        square = Square(side_length=2)
        pointwise_function = function_plot(lambda x: x**3, x_range=[-2, 2], y_range=[-8, 8])
        self.play(Write(square))
        self.play(ApplyFunctionToSubmobjects(pointwise_function, square))

class Scene4(Scene):
    def construct(self):
        text = Text("Text").next_to(ORIGIN, DOWN, buff=1)
        equation = MathTex("g(x) = sin(x)").next_to(text, DOWN, buff=1)
        self.play(Write(text), Write(equation))

class Scene5(Scene):
    def construct(self):
        square = Square(side_length=2)
        number = DecimalNumber(2, color=YELLOW)
        self.play(Write(square))
        self.play(run_time=3, animate=True, lag_ratio=0.05, function=lambda t: number.set_value(square.get_center()[0]))

class Scene6(Scene):
    def construct(self):
        shapes = [Circle(radius=1), Square(side_length=1), Rectangle(width=1, height=2)]
        pi_symbol = Pi()
        self.play(Create(shapes), Write(pi_symbol))
        self.play(
            SpiralIn(shapes, pi_symbol, radius=2, angle_radius=2, run_time=5),
            FadeOut(shapes),
            FadeOut(pi_symbol)
        )

class Scene7(Scene):
    def construct(self):
        triangle1 = Polygon(ORIGIN, UP*2, RIGHT*2, fill_color=YELLOW)
        triangle2 = Polygon(ORIGIN, DOWN*2, LEFT*2, fill_color=BLUE)
        triangle3 = Polygon(ORIGIN, UP*2, LEFT*2, fill_color=GREEN)
        self.play(Write(triangle1))
        self.play(Write(triangle2))
        self.play(Write(triangle3))
```

This code creates seven scenes, each with different animations as per your instructions. Make sure to have the manim library installed before running the code. You can run the scenes by creating an instance of each class and calling the `run()` method.