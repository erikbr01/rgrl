import tyro

from rgrl.eval_robomimic import EvalConfig, evaluate

if __name__ == "__main__":
    evaluate(tyro.cli(tyro.conf.PositionalRequiredArgs[EvalConfig]))
