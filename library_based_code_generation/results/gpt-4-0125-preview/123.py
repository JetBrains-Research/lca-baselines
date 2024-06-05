```python
from manim import *

class FirstScene(Scene):
    def construct(self):
        title = Text("Mathematical Visualization").scale(0.9)
        equation = MathTex(r"E = mc^2").scale(2)
        self.play(Write(title))
        self.play(Write(equation))
        self.wait(1)
        self.play(Transform(title, Text("Grid Transformation").scale(0.9)), FadeOut(equation))
        
        grid = NumberPlane()
        grid_title = Text("A Simple Grid").scale(0.9)
        self.play(Create(grid), Write(grid_title))
        self.wait(1)
        grid_transform_title = Text("Transformed Grid").scale(0.9)
        self.play(grid.animate.apply_function(lambda p: p + np.array([np.sin(p[1]), np.sin(p[0]), 0])),
                  Transform(grid_title, grid_transform_title))
        self.wait(2)

class SecondScene(Scene):
    def construct(self):
        square = Square()
        circle = Circle()
        self.play(Create(square))
        self.wait(1)
        self.play(Transform(square, circle))
        self.wait(1)
        self.play(FadeOut(square))

class ThirdScene(Scene):
    def construct(self):
        square = Square()
        self.play(Create(square))
        self.wait(1)
        self.play(square.animate.apply_function(lambda p: [p[0]*np.sin(p[1]), p[1]*np.cos(p[0]), 0]))
        self.wait(1)

class FourthScene(Scene):
    def construct(self):
        text = Text("Manim is Fun!").scale(0.9)
        equation = MathTex(r"\int_a^b f(x)\,dx").scale(2)
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
        shapes = VGroup(Circle(), Square(), Triangle()).arrange(RIGHT, buff=1)
        pi_symbol = MathTex(r"\pi").scale(5)
        self.play(FadeIn(shapes), FadeIn(pi_symbol))
        self.wait(1)
        self.play(Rotate(shapes, PI/2), Rotate(pi_symbol, PI/2))
        self.play(SpiralOut(shapes), SpiralOut(pi_symbol))
        self.wait(1)

class SeventhScene(Scene):
    def construct(self):
        triangle1 = Triangle().set_stroke(joint="round")
        triangle2 = Triangle().set_stroke(joint="bevel").next_to(triangle1, RIGHT)
        triangle3 = Triangle().set_stroke(joint="miter").next_to(triangle2, RIGHT)
        self.play(Create(triangle1), Create(triangle2), Create(triangle3))
        self.wait(2)
```