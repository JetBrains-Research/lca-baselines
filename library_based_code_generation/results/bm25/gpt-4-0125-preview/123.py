```python
from manim import *

class FirstScene(Scene):
    def construct(self):
        title = Text("Mathematical Introduction")
        equation = MathTex(r"E = mc^2")
        self.play(Write(title))
        self.play(Write(equation))
        self.wait(1)
        self.play(Transform(title, Text("Grid Transformation")), FadeOut(equation))
        grid = NumberPlane()
        grid_title = Text("This is a grid")
        self.play(Create(grid), Write(grid_title))
        self.wait(1)
        self.play(grid.animate.apply_function(lambda p: p + np.array([np.sin(p[1]), np.sin(p[0]), 0])),
                  Transform(grid_title, Text("Transformed Grid")))
        self.wait(2)

class SecondScene(Scene):
    def construct(self):
        square = Square()
        self.play(Create(square))
        self.wait(1)
        self.play(Transform(square, Circle()))
        self.wait(1)
        self.play(FadeOut(square))
        self.wait(1)

class ThirdScene(Scene):
    def construct(self):
        square = Square()
        self.play(square.animate.apply_function(lambda p: [p[0]*np.sin(p[1]), p[1]*np.cos(p[0]), 0]))
        self.wait(2)

class FourthScene(Scene):
    def construct(self):
        text = Text("A beautiful equation:")
        equation = MathTex(r"\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}")
        self.play(Write(text), Write(equation))
        self.wait(2)

class FifthScene(Scene):
    def construct(self):
        square = Square().to_edge(UP)
        decimal = DecimalNumber(0).to_edge(DOWN)
        self.add(square, decimal)
        self.play(square.animate.to_edge(DOWN))
        decimal.add_updater(lambda d: d.set_value(square.get_center()[1]))
        self.wait(2)
        decimal.clear_updaters()
        self.wait(1)

class SixthScene(Scene):
    def construct(self):
        shapes = VGroup(Circle(), Square(), Triangle())
        pi_symbol = MathTex(r"\pi").scale(5)
        self.play(FadeIn(shapes), FadeIn(pi_symbol))
        self.play(Rotate(shapes, PI/4), Rotate(pi_symbol, PI/4))
        self.play(FadeOut(shapes), FadeOut(pi_symbol))
        self.wait(1)

class SeventhScene(Scene):
    def construct(self):
        triangles = VGroup(
            Triangle().set_stroke(line_join=0),
            Triangle().next_to(TRIANGLE, RIGHT).set_stroke(line_join=1),
            Triangle().next_to(TRIANGLE, RIGHT, buff=2).set_stroke(line_join=2),
        )
        self.play(Create(triangles))
        self.wait(2)
```