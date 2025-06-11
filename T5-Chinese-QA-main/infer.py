from rich import print
from transformers import AutoTokenizer, T5ForConditionalGeneration

device = "cuda:0"
max_source_seq_len = 256
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/model_best/")
model = T5ForConditionalGeneration.from_pretrained("./checkpoints/model_best/")
model.to(device).eval()


def inference(qustion: str, context: str):
    input_seq = f"问题：{question}{tokenizer.sep_token}原文：{context}"
    inputs = tokenizer(
        text=input_seq,
        truncation=True,
        max_length=max_source_seq_len,
        padding="max_length",
        return_tensors="pt",
    )
    outputs = model.generate(input_ids=inputs["input_ids"].to(device))
    output = tokenizer.decode(
        outputs[0].cpu().numpy(), skip_special_tokens=True
    ).replace(" ", "")
    print(f'Q: "{qustion}"')
    print(f'C: "{context}"')
    print(f'A: "{output}"')


if __name__ == "__main__":
    # question = "2017年银行贷款基准利率"
    # context = "2017年基准利率4.35%。 从实际看,贷款的基本条件是: 一是中国大陆居民,年龄在60岁以下; 二是有稳定的住址和工作或经营地点; 三是有稳定的收入来源; 四是无不良信用记录,贷款用途不能作为炒股,赌博等行为; 五是具有完全民事行为能力。"
    question = "12306退票后多久能重新买票"
    context = "12306退票后当天可以购票, 在12306.cn网站购票后,请按以下方式办理退票: (1)没有换取纸质车票且不晚于开车前30分钟的,可以在12306.cn网站办理。 (2)已经换取纸质车票或者在开车前30分钟之内的,请携带购票时所使用的乘车人有效身份证件原件到车站售票窗口办理;居民身份证无法自动识读或者使用居民身份证以外的其他有效身份证件购票的,请提供订单号码(E+9位数字)。 (3)使用居民身份证购票且持居民身份证办理进站检票后,未乘车即出站的,请经车站确认后按规定办理。 (4)因伤、病或者承运人责任中途下车的,请凭列车长出具的客运记录和购票时所使用的乘车人有效身份证件原件在下车站按规定办理。 参考资料:https://kyfw.12306.cn/otn/gonggao/onlineRefund.html"
    inference(qustion=question, context=context)
