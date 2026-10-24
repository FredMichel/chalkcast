"""Codegen produces syntactically valid, expectation-matching Manim modules."""

from chalkcast.codegen import generate_module
from chalkcast.demo import build_demo_project, build_equation_scene


def test_demo_module_is_valid_python():
    _, scene = build_demo_project()
    code = generate_module(scene, "chalkboard")
    # Must compile as Python (proves no broken codegen).
    compile(code, "<demo>", "exec")


def test_demo_module_contains_expected_constructs():
    _, scene = build_demo_project()
    code = generate_module(scene, "chalkboard")
    assert "from manim import *" in code
    assert f"class {scene.class_name()}(MovingCameraScene):" in code
    assert "Axes(" in code             # the plot
    assert ".plot(lambda x: x**2" in code
    assert "self.play(Write(" in code  # a write step
    assert "Indicate(" in code         # the highlight step
    assert "background_color" in code  # theme applied


def test_equation_codegen():
    scene = build_equation_scene()
    code = generate_module(scene, "chalkboard")
    compile(code, "<eq>", "exec")
    assert "MathTex(" in code


def test_equation_backslashes_not_doubled():
    # regression: combining an r-prefix with repr() doubled every backslash,
    # turning \int into \\int (a LaTeX line break) and garbling the math.
    from chalkcast.model import Obj, Project

    tex = r"\int_0^1 x^2 \, dx"
    proj = Project(id="p")
    scene = proj.new_scene()
    scene.objects.append(Obj(id="equation_1", kind="equation", payload={"tex": tex}))
    code = generate_module(scene, "dark")
    assert "MathTex(r" not in code   # no raw-string prefix on the tex
    assert repr(tex) in code         # the correctly-escaped literal is present


def test_shape_and_camera_codegen():
    from chalkcast.model import Obj, Project

    project = Project(id="p")
    scene = project.new_scene()
    scene.objects.append(Obj(id="shape_1", kind="shape", position=(0, 0),
                             payload={"shape": "square", "size": 2}))
    scene.add_step("write", targets=["shape_1"])
    scene.add_step("camera_move", to="2@1,1", duration=1.0)
    code = generate_module(scene, "dark")
    compile(code, "<shapes>", "exec")
    assert "Square(side_length=2" in code
    assert "self.camera.frame.animate" in code
    assert "Create(shape_1)" in code  # shapes use Create, not Write
