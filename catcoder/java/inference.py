import os
import re

import backoff
from dotenv import load_dotenv

try:
    from openai import OpenAI, RateLimitError
except ImportError:
    print('Please install the openai package')
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

for env_path in ('.env', '../.env'):
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
        break

def remove_reasoning_markup(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return re.sub(r'^\s*(?:(?:</?think>|think>)\s*)+', '', text).lstrip()

def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {'0', 'false', 'no', 'off', ''}

def coerce_content(content) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get('text') or ''))
            else:
                parts.append(str(getattr(item, 'text', '') or ''))
        return ''.join(parts)
    return str(content)

def should_retry_without_extra_body(exc: Exception) -> bool:
    status_code = getattr(exc, 'status_code', None)
    message = str(exc).lower()
    if status_code in {400, 422}:
        return True
    return any(token in message for token in (
        'extra_body',
        'enable_thinking',
        'thinking_budget',
        'unsupported',
        'unknown parameter',
        'unrecognized',
    ))

class Model:
    def __init__(self, model_id: str, temp: float, top_p: float, **kwargs):
        self.model_id = model_id
        self.temp = temp
        self.top_p = top_p
        self.max_new_tokens = 512

    @property
    def info(self) -> str:
        return f'{self.model_id}_temp{self.temp}_topp{self.top_p}'
    
    def infer(self, prompt: str) -> str:
        raise NotImplementedError()

    @staticmethod
    def new(**kwargs) -> 'Model':
        provider = kwargs.pop('provider', None)
        if provider == 'openai':
            return OpenAIModel(**kwargs)
        elif provider == 'vllm-client':
            return VllmClientModel(**kwargs)
        elif provider == 'vllm':
            return VllmModel(**kwargs)
        else:
            if 'port' in kwargs:
                return VllmClientModel(**kwargs)
            elif 'model_path' in kwargs:
                return VllmModel(**kwargs)
            else:
                return OpenAIModel(**kwargs)
    
class OpenAIModel(Model):
    def __init__(self, model_id=None, temp=0.6, top_p=0.7, max_new_tokens=512,
                 stream=None, disable_thinking=None, thinking_budget=None,
                 client=None, **kwargs):
        if model_id is None:
            model_id = os.environ.get('OPENAI_MODEL')
            if model_id is None:
                raise ValueError('OPENAI_MODEL is not set. Please set it in .env or as an environment variable.')
        super().__init__(model_id, temp, top_p)
        self.max_new_tokens = max_new_tokens
        self.stream = env_bool('OPENAI_STREAM', False) if stream is None else stream
        self.disable_thinking = env_bool('OPENAI_DISABLE_THINKING', True) if disable_thinking is None else disable_thinking
        self.thinking_budget = int(os.environ.get('OPENAI_THINKING_BUDGET', '128')) if thinking_budget is None else thinking_budget
        self.client = client or OpenAI(api_key=os.environ['OPENAI_API_KEY'],
                                      base_url=os.environ['OPENAI_BASE_URL'])

    def _completion_kwargs(self, prompt: str) -> dict:
        request = {
            'model': self.model_id,
            'messages': [{'role': 'system', 'content': 'You are an expert at Java programming.'}, {'role': 'user', 'content': prompt}],
            'stream': self.stream,
            'temperature': self.temp,
            'top_p': self.top_p,
            'stop': ['[/CODE]', '/**'],
            'max_tokens': self.max_new_tokens,
        }
        if self.disable_thinking:
            extra_body = {'enable_thinking': False}
            if self.thinking_budget is not None:
                extra_body['thinking_budget'] = self.thinking_budget
            request['extra_body'] = extra_body
        return request

    def _extract_completion_text(self, completion) -> str:
        if self.stream:
            ans = ''
            for chunk in completion:
                if not getattr(chunk, 'choices', None):
                    continue
                delta = chunk.choices[0].delta
                ans += coerce_content(getattr(delta, 'content', None))
            return ans
        if not getattr(completion, 'choices', None):
            return ''
        message = completion.choices[0].message
        return coerce_content(getattr(message, 'content', None))
        
    @backoff.on_exception(backoff.expo, RateLimitError)
    def infer(self, prompt: str) -> str:
        task = self.client.chat.completions
        request = self._completion_kwargs(prompt)
        try:
            completion = task.create(**request)
        except Exception as exc:
            if 'extra_body' not in request or not should_retry_without_extra_body(exc):
                raise
            request.pop('extra_body', None)
            completion = task.create(**request)
        ans = self._extract_completion_text(completion)
        return remove_reasoning_markup(ans)

class VllmModel(Model):
    def __init__(self, model_id: str, model_path: str, 
                 temp=0.6, 
                 top_p=0.7,
                 dtype: str='auto',
                 gpu_ordinals: list[int]=None,
                 num_gpus: int=1,
                 gpu_memory_utilization: float=0.9,
                 quantization: str=None,
                 **kwargs
                 ):
        if LLM is None or SamplingParams is None:
            raise ImportError('Please install the vllm package to use VllmModel.')
        super().__init__(model_id, temp, top_p)
        if gpu_ordinals is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ordinals))
            num_gpus = min(num_gpus, len(gpu_ordinals))
        self.model = LLM(model=model_path,
                         tensor_parallel_size=num_gpus,
                         gpu_memory_utilization=gpu_memory_utilization,
                         quantization=quantization,
                         dtype=dtype,
                         )
        self.sampling_params = SamplingParams(temperature=temp,
                                              top_p=self.top_p,
                                              stop=['[/CODE]', '/**', 'public', 'private'],
                                              max_tokens=self.max_new_tokens,
                                              )

    def infer(self, prompt: str) -> str:
        response = self.model.generate(prompt, 
                                       self.sampling_params, use_tqdm=False)[0]
        return response.outputs[0].text

class VllmClientModel(Model):
    def __init__(self, model_id: str, port=3000, mock=False, temp=0.6, top_p=0.7, **kwargs):
        super().__init__(model_id, temp, top_p)
        if not mock:
            self.client = OpenAI(api_key=os.environ.get('VLLM_API_KEY', 'EMPTY'), base_url=f'http://localhost:{port}/v1')
            self._models = self.client.models.list()
            for model in self._models.data:
                if model.id == model_id:
                    self._model = model_id
                    break
            else:
                self._model = self._models.data[0].id
            print(f'user-side model_id: {model_id}, server-side model_id: {self._model}')

    def infer(self, prompt: str) -> str:
        task = self.client.completions
        completion = task.create(
            model=self._model,
            prompt=prompt,
            echo=False,
            stream=True,
            temperature=self.temp,
            top_p=self.top_p,
            stop=['[/CODE]', '/**', 'public', 'private'],
            max_tokens=self.max_new_tokens,
        )
        ans = ''
        for chunk in completion:
            if not chunk.choices:
                continue
            content = chunk.choices[0].text
            if content is not None:
                ans += content
        return ans
