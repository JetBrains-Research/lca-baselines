 ```python
from manim import *

class Scene1(Scene):
    def construct(self):
        title = Tex("Scene 1").to_edge(UP)
        eqn = MathTex(r"y = x^2").next_to(title, DOWN)
        self.play(Write(title), Write(eqn))
        self.play(Transform(title, Tex("Transformed Title")), FadeOut(eqn))

        grid = NumberPlane()
        grid_title = Tex("Grid").to_edge(UP)
        self.play(Create(grid_title), ShowCreation(grid))
        grid.apply_function(lambda p: p + np.array([0, 0.2, 0]))
        self.play(Transform(grid_title, Tex("Transformed Grid Title")))

class Scene2(Scene):
    def construct(self):
        square = Square()
        self.play(Create(square))
        circle = Circle().match_style(square)
        self.play(Transform(square, circle), RunTimeBar(run_time=2))
        self.play(FadeOut(circle))

class Scene3(Scene):
    def construct(self):
        square = Square()
        pointwise_fn = lambda point: max(point.y, 0) * RIGHT + min(point.y, 0) * LEFT
        new_square = square.apply_function(pointwise_fn)
        self.play(Create(square), RunTimeBar(run_time=2))
        self.play(Transform(square, new_square))

class Scene4(Scene):
    def construct(self):
        title = Tex("Scene 4").to_edge(UP)
        eqn = MathTex(r"y = \sin(x)").next_to(title, DOWN)
        self.play(Write(title), Write(eqn))

class Scene5(Scene):
    def construct(self):
        square = Square()
        decimal = DecimalNumber(decimal_number=0).next_to(square, RIGHT)
        self.play(Create(square), Write(decimal))
        self.play(square.animate.shift(2 * UP), decimal.animate.set_value(1))

class Scene6(Scene):
    def construct(self):
        shapes = VGroup(*[Circle() for _ in range(5)])
        pi_symbol = Tex("\\pi").scale(3)
        self.play(Create(shapes), Write(pi_symbol))
        self.play(shapes.animate.spiral_in(run_time=3), FadeOut(pi_symbol))

class Scene7(Scene):
    def construct(self):
        tri1 = Polygon(ORIGIN, 3 * RIGHT, 3 * UP)
        tri2 = Polygon(ORIGIN, 3 * LEFT, 3 * UP)
        tri3 = Polygon(ORIGIN, 3 * RIGHT, 3 * DOWN)
        tri1.set_stroke(width=5)
        tri2.set_stroke(width=5)
        tri3.set_stroke(width=5)
        self.play(Create(tri1), Create(tri2), Create(tri3))
```
This code defines 7 different scenes using the Manim library. Each scene has its own unique animation or transformation.