  Here is the Python code based on the given instructions:
```
from manim import *

class Scene1(Scene):
    def construct(self):
        title = Tex("LaTeX Title")
        equation = MathTex("f(x) = x^2 + 1")
        self.play(Write(title), Write(equation))
        self.play(FadeOut(title), FadeOut(equation))

        grid = Grid(color=WHITE)
        grid_title = Tex("Grid")
        self.play(Write(grid), Write(grid_title))
        self.play(ApplyFunction(lambda x: x**2, grid), FadeOut(grid_title))

class Scene2(Scene):
    def construct(self):
        square = Square()
        self.play(Transform(square, Circle()))
        self.play(FadeOut(square))

class Scene3(Scene):
    def construct(self):
        square = Square()
        self.play(ApplyFunction(lambda x: x**2, square))
        self.play(FadeOut(square))

class Scene4(Scene):
    def construct(self):
        text = Text("Text")
        equation = MathTex("f(x) = x^2 + 1")
        self.play(Write(text), Write(equation))
        self.play(FadeOut(text), FadeOut(equation))

class Scene5(Scene):
    def construct(self):
        square = Square()
        decimal = DecimalNumber(0)
        self.play(Write(square), Write(decimal))
        self.play(FadeOut(square), FadeOut(decimal))

class Scene6(Scene):
    def construct(self):
        shapes = [Square(), Circle(), Triangle()]
        pi_symbol = MathTex("\\pi")
        self.play(Write(shapes), Write(pi_symbol))
        self.play(SpiralIn(shapes), FadeOut(pi_symbol))

class Scene7(Scene):
    def construct(self):
        triangles = [Triangle(color=RED), Triangle(color=GREEN), Triangle(color=BLUE)]
        self.play(Write(triangles))
        self.play(FadeOut(triangles))
```