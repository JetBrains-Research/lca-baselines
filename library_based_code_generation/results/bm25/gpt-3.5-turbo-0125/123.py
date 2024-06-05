```python
from manim import *

class FirstScene(Scene):
    def construct(self):
        title = Tex("Manim Scenes")
        equation = MathTex("e^{i\pi} + 1 = 0")
        self.play(Write(title))
        self.play(FadeIn(equation))
        self.wait(1)
        self.play(Transform(title, equation))
        self.play(FadeOut(equation))

        grid = NumberPlane()
        grid_title = Tex("Grid")
        self.play(Create(grid))
        self.play(Write(grid_title))
        self.play(grid.apply_function, lambda p: p + np.array([np.sin(p[1]), np.cos(p[0]), 0]))
        self.play(Transform(grid_title, equation))

class SecondScene(Scene):
    def construct(self):
        square = Square()
        circle = Circle()
        self.play(Create(square))
        self.play(Transform(square, circle))
        self.play(FadeOut(square))

class ThirdScene(Scene):
    def construct(self):
        square = Square()
        square.apply_function(lambda p: p + np.array([np.sin(p[1]), np.cos(p[0]), 0]))
        self.play(Create(square))

class FourthScene(Scene):
    def construct(self):
        text = Text("Manim is awesome!")
        equation = MathTex("x^2 + y^2 = r^2")
        self.play(Write(text))
        self.play(FadeIn(equation))

class FifthScene(Scene):
    def construct(self):
        square = Square()
        decimal = DecimalNumber(0)
        decimal.add_updater(lambda d: d.set_value(square.get_center()[0]))
        decimal.add_updater(lambda d: d.next_to(square, UP))
        self.play(Create(square))
        self.play(Write(decimal))

class SixthScene(Scene):
    def construct(self):
        shapes = [Circle(), Square(), Triangle()]
        pi_symbol = MathTex("\\pi")
        self.play(*[Create(shape) for shape in shapes])
        self.play(Create(pi_symbol))
        self.play(*[ApplyMethod(shape.rotate, TAU) for shape in shapes])
        self.play(FadeOut(*shapes), FadeOut(pi_symbol))

class LastScene(Scene):
    def construct(self):
        triangle1 = Triangle()
        triangle2 = Triangle()
        triangle3 = Triangle()
        triangle1.shift(LEFT)
        triangle3.shift(RIGHT)
        self.play(Create(triangle1), Create(triangle2), Create(triangle3))
```