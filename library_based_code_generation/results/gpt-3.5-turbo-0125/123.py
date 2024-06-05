```python
from manim import *

class FirstScene(Scene):
    def construct(self):
        title = Tex("Manim Library")
        equation = MathTex("e^{i\pi} + 1 = 0")
        self.play(Write(title))
        self.play(Write(equation))
        self.play(Transform(title, title.copy().shift(UP*3)), FadeOut(equation))

        grid = NumberPlane()
        grid_title = Tex("Grid")
        self.play(Create(grid), Write(grid_title))
        self.play(grid.apply_function, lambda p: p + np.array([np.sin(p[1]), np.cos(p[0]), 0]), run_time=3)
        self.play(Transform(grid_title, grid_title.copy().shift(DOWN*3))

class SecondScene(Scene):
    def construct(self):
        square = Square()
        circle = Circle()
        self.play(Create(square))
        self.play(Transform(square, circle), FadeOut(circle))

class ThirdScene(Scene):
    def construct(self):
        square = Square()
        square.apply_function(lambda p: p + np.array([np.sin(p[1]), np.cos(p[0]), 0]))
        self.play(Create(square))

class FourthScene(Scene):
    def construct(self):
        text = Text("Hello, Manim!")
        equation = MathTex("x^2 + y^2 = r^2")
        self.play(Write(text), Write(equation))

class FifthScene(Scene):
    def construct(self):
        square = Square()
        decimal_number = DecimalNumber(0)
        decimal_number.add_updater(lambda d: d.set_value(square.get_center()[0]))
        decimal_number.add_updater(lambda d: d.move_to(square.get_center()))
        self.play(Create(square), Write(decimal_number))

class SixthScene(Scene):
    def construct(self):
        shapes = [Circle(), Square(), Triangle()]
        pi_symbol = MathTex("\\pi")
        self.play(*[Create(shape) for shape in shapes], Write(pi_symbol))
        self.play(*[ApplyMethod(shape.rotate, 2*PI) for shape in shapes], FadeOut(pi_symbol))

class LastScene(Scene):
    def construct(self):
        triangle1 = Triangle()
        triangle2 = Triangle().shift(LEFT*3)
        triangle3 = Triangle().shift(RIGHT*3)
        self.play(Create(triangle1), Create(triangle2), Create(triangle3))
```
