from text.symbols import symbols

_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def cleaned_text_to_sequence(cleaned_text):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
  return sequence
#这段代码实现了将文本字符串转换为对应符号的 ID 序列的功能。它导入了一个名为 symbols 的列表，其中包含了一些符号。
#在 _symbol_to_id 字典中，将每个符号映射为其在 symbols 中的索引。函数 cleaned_text_to_sequence 接受一个经过清理的文本字符串，
#并将其中的每个符号转换为对应的 ID，最终返回一个 ID 列表。具体来说，它遍历了字符串中的每个符号，
#通过 _symbol_to_id 查找该符号在 symbols 中的索引，并将其加入到序列中。