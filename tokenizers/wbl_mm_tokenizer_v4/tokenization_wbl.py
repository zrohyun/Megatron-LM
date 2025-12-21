import re
import numpy as np

from typing import Union, List
from transformers import PreTrainedTokenizerFast

# \n, \t 이후 단일 공백인 경우만 매칭
_TN_FIX_RE = re.compile(r"(\n|\t) (?! )")

class WBLTokenizer(PreTrainedTokenizerFast):
    def decode(
        self,
        token_ids: Union[int, List[int], List[List[int]], np.ndarray, "torch.Tensor"],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> Union[str, List[str]]:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        text = super().decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            **kwargs,
        )
        # 맨 앞의 token ids가 134~156 이면 stripped whitespace를 복원
        if len(token_ids) > 0 and token_ids[0] >= 134 and token_ids[0] <= 156:
            text = " " + text

        # \n, \t 다음의 공백 위치가 multiple whitespace가 아닌 경우에는 whitespace를 하나 제거
        # 157~176 이 \n, ... \t\t 에 해당.
        text = _TN_FIX_RE.sub(r"\1", text)

        return text