"""Scope one-use permissions and expand fixed macros into existing tools."""

from brains.base import Decision


def command(tool, **args):
    return {"tool": tool, "args": args}


def scope(state, action):
    return (state.frame.id, state.phase,
            state.target.model_dump_json() if state.target else None, action)


def authorize(state, action, required):
    for request in required:
        if not any(s["kind"] == request.kind and s["direction"] == request.direction
                   for s in state.loaded):
            raise RuntimeError(f"Missing requested evidence: {request}")
    state.permission = scope(state, action)
    state.can_act = True


def issue(state, commands, note, *, action=None, fixed=False):
    if not fixed and (not state.can_act or state.permission != scope(state, action)):
        raise RuntimeError("No permission for this frame/target/phase/action")
    if state.pending_commands:
        raise RuntimeError("Must receive execution feedback before issuing more actions")
    state.pending_commands = commands
    state.can_act = False
    state.permission = None
    return [Decision(note, entry["tool"], entry["args"]) for entry in commands]


def grasp():
    return [command("open_gripper"), command("arm_pose", pose="reach"),
            command("close_gripper"), command("arm_pose", pose="carry")]


def drop():
    return [command("arm_pose", pose="drop"), command("open_gripper"), command("observe")]
