from configs import *


deepseek_test_1 = "<|ref|>text<|/ref|><|det|>[[120, 99, 930, 333]]<|/det|> 8 ཀུགས་ཀྱི་ཚོགས་ཀྱི་ཚོ"
deepseek_test_2 = "<|ref|>text<|/ref|><|det|>[[103, 0, 930, 333]]<|/det|>\n\n๑\n\n๑๖๖๖๖๖๖๖๖๖๖๖๖๖๖๖๗๖๖๖๖๖๖๖๗๖๖๗๖ "
deepseek_test_3 = """<|ref|>sub_title<|/ref|><|det|>[[390, 150, 924, 333]]<|/det|>
بۇنداق ئۇرنال ياشلارغا زىيانلىق."""
print(extract_deepseek(deepseek_test_3))

dots_test_1 = "متأسفم، در حال حاضر اتاق خالی نداریم. #"
# print(extract_default(dots_test_1))

firered_test_1 = "# هي كاري به عقavid مذهبی من نداشته باش ,باشه ?"
firered_test_2 = "**تقریباً تاسی دیقق دیگر بندی رسم.**"
firered_test_3 = "**Converting to Markdown** \n\nI'm now satisfied with the Markdown output. I've double-checked the text, ensuring accurate transcription and formatting. The output is ready. \n Hi"
firered_test_4 = "** converting the text to Markdown format** \n \n \n # ฉันรู้สึกว่าฉันรู้สึกว่าฉันรู้สึกว่าฉันรู้สึกว่าฉันรู้สึกว่าฉันรู้สึกว่าฉันรู้สึกว่าฉันรู้สึกว่าฉันรู้สึกว่าฉันรู้สึกว่าฉันรู้สึกว่าฉันรู้สึกว่าฉันรู้สึกว่าฉันรู้สึกว่าฉันรู้สึกว่าฉันรู้สึกว่าฉันรู้สึกว่าฉันรู้สึกว่าฉันรู้สึกว่าฉันรู้สึกว่าฉันรู้สึกว่าฉันรู้สึกว่าฉันรู้สึกว่าฉันรู้สึกว่าฉันรู้สึกว่าฉันรู้สึกว่าฉันรู้สึกว่าฉั"
# print(extract_firered(firered_test_4))

lighton_test_1 = "$\mathcal{J}\mathcal{J}\mathcal{E}\mathcal{Y}\mathcal{L}\mathcal{Z}\mathcal{T}\mathcal{O}\mathcal{J}$"
lighton_test_2 = """"$\Sigma 0 \\text{J} 4 4 \\text{J} 2 0 \\text{J} 4$"""
lighton_test_3 = "$\Sigma 0 \text{J} 4 4 \text{J} 2 0 \text{J} 4$"
lighton_test_4 = "$\text{تقریماً ماسی وقیق دیگر به بندر می رسمیم.}$"
lighton_test_5 = "$\text{מִשְׁלָה 35מָכָל יְהוָה יְהוָה כָּל יְהוָה יְהוָה יְהוָה}$\n\n$\text{כָּל יְהוָה יְהוָה יְהוָה יְהוָה יְהוָה}$"
lighton_test_6 = "$\text{τρτ} \text{ hωλκωρδωhωu} \text{ ψρητητηρ} \text{ μωh} \text{ ρτρh} \text{ ,}$"
# print(extract_lighton(lighton_test_6))


paddleocr_test_1 = "The provided image is a graphic design and does not contain any chart, graph, or data to be converted into a table.\nHi"
# print(extract_paddleocr(paddleocr_test_1))

qwen3_test_1 = """```markdown
ચાલુ અને અનુભવિત સંસ્થાઓ માટે એક અનુભવિત
```"""


qwen3_test_2 = """\( \text{मिट्टी के द्वारा विकसित उपकरण} \)"""

qwen3_test_3 = """
The provided image contains text written in the Malayalam script. Since there are no mathematical formulas, tables, or figures to convert, and the instruction is to convert the text into Markdown format, the text can be directly presented as-is in Markdown.

Here is the text from the image:

```markdown
ഒരു ക്രിസ്തീയ ക്രിസ്തുവിന്റെ കാലം അനുഭവിക്കുന്ന വിഭാഗം
```

"""

qwen3_test_4 = """
The provided image contains text written in the Malayalam script. Since there are no mathematical formulas, tables, or figures to convert, and the instruction is to convert the text into Markdown format, the text can be directly presented as-is in Markdown.

Here is the text from the image:

```markdown
ഒരു ക്രിസ്തീയ ക്രിസ്തുവിന്റെ കാലം അനുഭവിക്കുന്ന വിഭാഗം

"""

qwen3_test_5 = """
The provided image contains text written in the Malayalam script. Since there are no mathematical formulas, tables, or figures to convert, and the instruction is to convert the text into Markdown format, the text can be directly presented as-is in Markdown.

Here is the text from the image:

```
ഒരു ക്രിസ്തീയ ക്രിസ്തുവിന്റെ കാലം അനുഭവിക്കുന്ന വിഭാഗം

"""

qwen3_test_6 = """
The provided image contains text written in the Malayalam script. Since there are no mathematical formulas, tables, or figures to convert, and the instruction is to convert the text into Markdown format, the text can be directly presented as-is in Markdown.

Here is the text from the image:

ഒരു ക്രിസ്തീയ ക്രിസ്തുവിന്റെ കാലം അനുഭവിക്കുന്ന വിഭാഗം

"""

qwen3_test_7 = """\(\mathbf{B}\) \(\mathbf{B}\) \(\mathbf{B}\) \(\mathbf{B}\) \(\mathbf{B}\) \(\mathbf{B}\) \(\mathbf{B}\) \(\mathbf{B}\) \(\mathbf{B}\) \ """
qwen3_test_8 = """\( \frac{1}{2} \)"""
qwen3_test_9 = """This is an inline formula \( E = mc^2 \)"""
qwen3_test_10 = """\( \dot{m} \, \kappa \tau \, \chi \iota \nu \, \rho \dot{\iota} \, \dot{\iota} \, \eta \iota \, \nu \dot{\iota} \, \eta \, \nu \dot{\iota} \, \\alpha \, \kappa \, \psi \)"""
qwen3_test_11 = """\( \\frac{1}{2} \\theta \\frac{d}{dt} \\theta + \\frac{1}{2} \phi \\frac{d}{dt} \phi = 0\)"""
qwen3_test_12 = """\( \\text{๑๘๐๘} \)"""
qwen3_test_13 = """วชิร์ ศิริพัฒน์ ที่ ทุ่มเท หัวใจ ต่อเนื่อง"""
# print(extract_qwen3(qwen3_test_13))