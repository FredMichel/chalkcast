"""Authoring-time validation, in-place editing, and expression safety."""

import pytest

pytest.importorskip("mcp", reason="mcp extra not installed")

import chalkcast.server as srv  # noqa: E402
from chalkcast.expr import ExprError, validate  # noqa: E402


@pytest.fixture()
def scene():
    srv.open_project()
    return srv.add_scene("t")["result"]["scene_id"]


# ---- expressions --------------------------------------------------------------

@pytest.mark.parametrize("expr", ["sin(x)", "np.sin(x)**2", "x**3 - x", "sign(sin(pi*x/5))",
                                  "4/pi*sum(sin((2*k+1)*x)/(2*k+1) for k in range(5))",
                                  "1/x", "sqrt(x)", "exp(-x**2/2)", "where(x>0, x, 0)"])
def test_valid_expressions(expr):
    validate(expr)


@pytest.mark.parametrize("expr,needle", [
    ("foo(x)", "unknown name 'foo'"),
    ("__import__('os').system('ls')", "only np.<function> attributes"),
    ("x.__class__", "only np.<function>"),
    ("np.nonexistent(x)", "not a numpy function"),
    ("sin(x", "syntax error"),
    ("", "empty"),
    ("sin(x, 3, 4)", "fails to evaluate"),
])
def test_invalid_expressions_rejected(expr, needle):
    with pytest.raises(ExprError, match=needle.replace("(", r"\(").replace(".", r"\.")):
        validate(expr)


def test_parameters_in_expressions():
    validate("sin(x - t)", params={"t"})
    with pytest.raises(ExprError):
        validate("sin(x - t)")


# ---- fail fast at authoring time -------------------------------------------------

def test_add_plot_rejects_bad_expression_immediately(scene):
    r = srv.add_plot(functions=[{"expr": "sinus(x)"}])
    assert r["ok"] is False and "unknown name 'sinus'" in r["error"]


def test_add_plot_without_functions_is_bare_axes(scene):
    # no silent default parabola
    r = srv.add_plot()
    pid = r["result"]["object_id"]
    obj = srv.session.get_project().get_scene(scene).get_obj(pid)
    assert obj.payload["functions"] == []


def test_bad_color_and_range_rejected(scene):
    assert srv.add_text("a", color="ROUGE")["ok"] is False
    assert srv.add_plot(functions=[{"expr": "x"}], x_range=[3, 1])["ok"] is False
    assert srv.add_shape(shape="star")["ok"] is False


def test_add_step_validates_targets_and_kinds(scene):
    t = srv.add_text("a")["result"]["object_id"]
    assert srv.add_step("write", targets=["text_99"])["ok"] is False
    assert srv.add_step("explode", targets=[t])["ok"] is False
    assert srv.add_step("transform", targets=[t])["ok"] is False          # no destination
    assert srv.add_step("camera_move")["ok"] is False                     # no zoom
    assert srv.add_step("tween", param="t", value=1)["ok"] is False       # undeclared
    assert srv.add_step("write", targets=[t])["ok"] is True


def test_tween_and_with_previous(scene):
    assert srv.add_parameter("t", 0.0)["ok"] is True
    assert srv.add_parameter("sin", 0.0)["ok"] is False  # shadows a math function
    p = srv.add_plot(functions=[{"expr": "sin(x - t)"}], x_range=[0, 10], y_range=[-1, 1])
    assert p["ok"] is True
    pid = p["result"]["object_id"]
    d = srv.add_shape("dot", plot_id=pid, at_x="t", at_y="0")["result"]["object_id"]
    assert srv.add_step("write", targets=[pid])["ok"]
    assert srv.add_step("write", targets=[d], with_previous=True)["ok"]
    assert srv.add_step("tween", param="t", value=6.28, duration=4)["ok"]


# ---- in-place editing ------------------------------------------------------------

def test_update_object_changes_only_given_fields(scene):
    eq = srv.add_equation(r"E=mc^2", x=1, y=2)["result"]["object_id"]
    r = srv.update_object(eq, x=-1, scale=0.8)
    assert r["ok"] and set(r["result"]["changed"]) == {"position", "scale"}
    obj = srv.session.get_project().get_scene(scene).get_obj(eq)
    assert obj.position == (-1, 2) and obj.scale == 0.8 and obj.payload["tex"] == "E=mc^2"
    assert srv.update_object(eq, font_size=30)["ok"] is False  # not a text
    assert srv.update_object(eq)["ok"] is False                # nothing to change


def test_update_plot_functions_validated(scene):
    pid = srv.add_plot(functions=[{"expr": "x"}])["result"]["object_id"]
    assert srv.update_object(pid, functions=[{"expr": "bad(x)"}])["ok"] is False
    assert srv.update_object(pid, functions=[{"expr": "x**2"}], y_range=[0, 25])["ok"] is True


def test_remove_object_requires_cascade_when_used(scene):
    pid = srv.add_plot(functions=[{"expr": "x"}], x_range=[0, 1], y_range=[0, 1])["result"]["object_id"]
    aid = srv.add_area(pid, 0, 1)["result"]["object_id"]
    srv.add_step("write", targets=[pid])
    srv.add_step("write", targets=[aid])
    r = srv.remove_object(pid)
    assert r["ok"] is False and r["result"]["dependent_objects"] == [aid]
    r = srv.remove_object(pid, cascade=True)
    assert r["ok"] and set(r["result"]["removed_objects"]) == {pid, aid}
    assert r["result"]["removed_steps"] == [1, 2]
    sc = srv.session.get_project().get_scene(scene)
    assert sc.objects == [] and sc.steps == []


def test_remove_step_renumbers(scene):
    t = srv.add_text("a")["result"]["object_id"]
    srv.add_step("write", targets=[t])
    srv.add_step("wait", duration=1)
    srv.add_step("highlight", targets=[t])
    r = srv.remove_step(2)
    assert r["ok"] and [s["index"] for s in r["result"]["steps"]] == [1, 2]
    assert [s["kind"] for s in r["result"]["steps"]] == ["write", "highlight"]
