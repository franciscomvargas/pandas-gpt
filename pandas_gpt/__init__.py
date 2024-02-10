import pandas as pd
import os

from threading import Thread

verbose = False # Override default setting with `pandas_gpt.verbose = True`
mutable = False # Override default setting with `pandas_gpt.mutable = True`

model = 'gpt-3.5-turbo'
completion_config = {}

template = '''
Write a Python function `process({arg_name})` which takes the following input value:

{arg_name} = {arg}

This is the function's purpose: {goal}
This fuction's outputs MUST be exported to `{out_dir}` folder with the respective file extension. 
Attention:
 * If ploting, instead of showing it, you must only export it.
'''

_ask_cache = {}

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

class Ask:
  def __init__(self, *, verbose=None, mutable=None, out_dir=None):
    self.verbose = verbose if verbose is not None else globals()['verbose']
    self.mutable = mutable if mutable is not None else globals()['mutable']
    self.out_dir = out_dir if out_dir is not None else os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

  @staticmethod
  def _fill_template(template, **kw):
    import re
    from textwrap import dedent
    result = dedent(template.lstrip('\n').rstrip())
    for k, v in kw.items():
      result = result.replace(f'{{{k}}}', v)
    m = re.match(r'\{[a-zA-Z0-9_]*\}', result)
    if m:
      raise Exception(f'Expected variable: {m.group(0)}')
    return result

  def _get_prompt(self, goal, arg):
    if isinstance(arg, pd.DataFrame) or isinstance(arg, pd.Series):
      import io
      buf = io.StringIO()
      arg.info(buf=buf)
      arg_summary = buf.getvalue()
    else:
      arg_summary = repr(arg)
    arg_name = 'df' if isinstance(arg, pd.DataFrame) else 'index' if isinstance(arg, pd.Index) else 'data'

    return self._fill_template(template, arg_name=arg_name, arg=arg_summary.strip(), goal=goal.strip(), out_dir=self.out_dir)

  def _run_prompt(self, prompt):
    import openai
    cache = _ask_cache
    completion = cache.get(prompt) or openai.chat.completions.create(
      model=model,
      messages=[
        dict(role='system', content='Write the function in a Python code block with all necessary imports and no example usage.'),
        dict(role='user', content=prompt),
      ],
      **completion_config,
    )
    cache[prompt] = completion
    return completion.choices[0].message.content

  def _extract_code_block(self, text):
    import re
    pattern = r'```(\s*(py|python)\s*\n)?([\s\S]*?)```'
    m = re.search(pattern, text)
    if not m:
      return text
    return m.group(3)
  
  def _eval(self, source, *args):
    source = source.replace("plt.style.use('seaborn')", "plt.style.use('classic')")
    _args_ = args
    scope = dict(_args_=args)
    exec(self._fill_template('''
      {source}
      _result_ = process(*_args_)
    ''', source=source), scope)
    _result = scope['_result_']
    print(f"[ DEBUG ] type: {type(_result)}")
    # print(f"[ DEBUG ]  res: {_result}")
    return _result

  def _code(self, goal, arg):
    prompt = self._get_prompt(goal, arg)
    result = self._run_prompt(prompt)
    if self.verbose:
      print()
      print(result)
    return self._extract_code_block(result)

  def code(self, *args):
    print(self._code(*args))

  def prompt(self, *args):
    print(self._get_prompt(*args))

  def __call__(self, goal, *args):
    source = self._code(goal, *args)
    # Create the thread that will start your function
    from threading import Thread
    eval_thread = Thread(target=self._eval, args=(source, *args))
    # Start the thread
    eval_thread.start()
    # Join your thread with the execution time you want
    eval_thread.join() #120
    # return self._eval(source, *args)
    return None


@pd.api.extensions.register_dataframe_accessor('ask')
@pd.api.extensions.register_series_accessor('ask')
@pd.api.extensions.register_index_accessor('ask')
class AskAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        pass

    def _ask(self, **kw):
      return Ask(**kw)

    def _data(self, **kw):
      if not mutable and not kw.get('mutable') and hasattr(self._obj, 'copy'):
        return self._obj.copy() # TODO: possibly `deep=False`
      return self._obj

    def __call__(self, goal, *args, **kw):
        ask = self._ask(**kw)
        data = self._data(**kw)
        return ask(goal, data, *args)

    def code(self, goal, *args, **kw):
        ask = self._ask(**kw)
        data = self._data(**kw)
        return ask.code(goal, data, *args)

    def prompt(self, goal, *args, **kw):
        ask = self._ask(**kw)
        data = self._data(**kw)
        return ask.prompt(goal, data, *args)
