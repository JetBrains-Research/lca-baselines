 Here is the Python code using the manim library as per your instructions:

```python
from manim import *

class Scene1(Scene):
    def construct(self):
        title = MathTitle("Title")
        equation = MathTex("f(x) = x^2")
        self.play(Write(title), Write(equation))
        self.wait(1)
        self.play(Transform(title, title.copy().next_to(equation, DOWN, buff=0.5)), FadeOut(equation))
        self.wait(1)

        grid = grid1.Grid(x_range=[0, 10, 1], y_range=[0, 10, 1])
        grid_title = Text("Grid").next_to(grid, UP)
        self.play(Write(grid_title))
        self.play(grid.animate.shift(RIGHT * 1), run_time=2)
        self.play(grid.apply_function(lambda x, y: x**2), run_time=2)
        self.play(Transform(grid_title, grid_title.copy().next_to(grid, DOWN, buff=0.5)))
        self.wait(1)

class Scene2(Scene):
    def construct(self):
        square = Square()
        self.play(Write(square))
        self.play(Transform(square, Circle()))
        self.wait(1)
        self.play(FadeOut(square))

class Scene3(Scene):
    def construct(self):
        square = Square()
        self.play(Write(square))
        square.set_points_expr(lambda x, y: x**2 + y**2)
        self.wait(1)

class Scene4(Scene):
    def construct(self):
        text = Text("Text").next_to(UP, 2)
        equation = MathTex("g(x) = sin(x)")
        self.play(Write(text), Write(equation))
        self.wait(1)

class Scene5(Scene):
    def construct(self):
        square = Square()
        number = DecimalNumber(0, color=YELLOW)
        self.play(Write(square), Write(number))
        square.move_to(ORIGIN)
        number.move_to(ORIGIN + RIGHT * 5)
        self.play(square.animate.move_to, ORIGIN + RIGHT * 5, run_time=2)
        number.set_value(square.get_center()[0])
        self.wait(1)

class Scene6(Scene):
    def construct(self):
        shapes = [Circle(radius=1), Square(fill_opacity=0), Rectangle(height=2, width=3)]
        pi = Text("π").scale(3)
        self.play(Write(shapes), Write(pi))
        self.play(*[shapes[i].animate.rotate_about(ORIGIN, i * math.pi / 4) for i in range(len(shapes))], run_time=2)
        self.play(*[FadeOut(obj) for obj in shapes + [pi]])

class Scene7(Scene):
    def construct(self):
        triangle1 = Polygon(point1=ORIGIN, point2=UP * 3, point3=RIGHT * 3)
        triangle2 = Polygon(point1=ORIGIN, point2=UP * 3, point3=LEFT * 3)
        triangle3 = Polygon(point1=ORIGIN, point2=UP * 2, point3=RIGHT * 2)
        self.play(Write(triangle1), Write(triangle2), Write(triangle3))
        self.wait(1)
```

This code creates seven scenes, each with different animations as per your instructions. Make sure to have the manim library installed before running the code. You can run the scenes by creating an instance of each class and calling the `run()` method.