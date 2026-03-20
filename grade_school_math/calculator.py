from contextlib import contextmanager
import signal
import torch as th
'''
    解决模型算术计算错误问题：
        1. 带超时的公式求值（eval_with_timeout），避免恶意计算；
        2. 采样时触发计算器（检测<</=，替换模型的算术结果）
'''

# taken from
# https://stackoverflow.com/questions/492519/timeout-on-a-function-call
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    # signal.signal(signal.SIGALRM, timeout_handler)
    '''
        signal 模块中，signal.alarm() 函数是 Unix/Linux/macOS 系统专属的
        signal.alarm(): 一个用于设置定时器的系统级调用
        核心作用: 防止模型在调用计算器时陷入死循环或执行时间过长的恶意计算
    '''
    # signal.alarm(duration)
    # yield
    # signal.alarm(0)


def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            return eval(formula)
    except Exception as e:
        # signal.alarm(0)
        print(f"Warning: Failed to eval {formula}, exception: {e}")
        return None


def use_calculator(sample):
    if "<<" not in sample:
        return None

    parts = sample.split("<<")
    remaining = parts[-1]
    if ">>" in remaining:
        return None
    if "=" not in remaining:
        return None
    lhs = remaining.split("=")[0]
    lhs = lhs.replace(",", "")
    if any([x not in "0123456789*+-/.()" for x in lhs]):
        return None
    return eval_with_timeout(lhs)


def sample(model, qn, tokenizer, device, sample_len):
    # Inefficient version of calculator sampling -- no batches, doesn't
    # cache activations from previous tokens
    EQUALS_TOKENS = set([28, 796, 47505])

    for _ in range(sample_len):
        with th.no_grad():
            toks = tokenizer([qn], padding=False, return_tensors="pt").to(device)
            orig_len = toks["input_ids"].shape[1]

            out = model.generate(
                **toks, max_length=orig_len + 1, pad_token_id=model.config.eos_token_id
            )
            text = tokenizer.batch_decode(out)[0]

            if out[0, -1].item() in EQUALS_TOKENS:
                answer = use_calculator(text)
                if answer is not None:
                    print("Triggered calculator, answer", answer)
                    text = text + str(answer) + ">>"

            qn = text
    return qn
