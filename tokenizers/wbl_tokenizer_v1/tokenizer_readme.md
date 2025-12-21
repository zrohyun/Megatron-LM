# WBL Tokenizer

## 추가된 토큰 (Reserved Token 교체)

- **<tool_start>**
- **<tool_end>**
- **<tools>**
- **</tools>**
- **<tool_call>**
- **</tool_call>**
- **<think>**
- **</think>**
- **<|role_start|>**
- **<|role_end|>**
- **<|START|>**
- **<|END|>**

### **Special Tokens**

- **<|role_start|>**
- **<|role_end|>**
- **<|START|> → (BOS Token for Instruction Tuning)**
- **<|END|> → (EOS Token for Instruction Tuning**

## 특징

- Qwen/Qwen3-235B-A22B-Thinking-2507 기준으로 토크나이저 수정
- assistant 발화는 </think>기준으로 split하여 전자는 rationale로 삽입
- tool call의 경우 별도의 think를 하지 않도록 add_generation_prompt=True시 </think>  발화 생성 전 고정
- *apply_chat_template(~~, tools=~~ )*  인 경우, Tool calling 인식
- Multi-turn시 이전 턴의 rationale는 반영되지 않음
- 편리한 데이터 토크나이징을 위해, 마지막 발화가 assistant 발화로 끝난 경우 <|END|> 자동 삽입
- 기존 <s>, </s> 는 BOS, EOS로 유지, <|START|>, <|END|> 각각 BOS, EOS에 추가

## 예시

- Single Turn - system, user (add_generation_prompt=True)
    
    ```python
    chat = [
        {"role":"system", "content":"You are a helpful assistant."},
        {"role":"user", "content":"Hello, who are you?"},
    ]
    ```
    
    ```python
    <|role_start|>system<|role_end|>
    You are a helpful assistant.
    <|role_start|>user<|role_end|>
    Hello, who are you?
    <|role_start|>assistant<|role_end|>
    <think>
    
    ```
    
- Single Turn - system, user, assistant (add_generation_prompt=False)
    
    ```python
    chat = [
        {"role":"system", "content":"You are a helpful assistant."},
        {"role":"user", "content":"Hello, who are you?"},
        {"role":"assistant", "content":"<think>User is asking about my identity.</think>I am a helpful assistant."}
    ]
    ```
    
    ```python
    <|role_start|>system<|role_end|>
    You are a helpful assistant.
    <|role_start|>user<|role_end|>
    Hello, who are you?
    <|role_start|>assistant<|role_end|>
    <think>
    User is asking about my identity.
    </think>
    
    I am a helpful assistant.<|END|>
    ```
    
- Multi Turn - system, user (add_generation_prompt=True)
    
    ```python
    chat = [
        {"role":"system", "content":"You are a helpful assistant."},
        {"role":"user", "content":"Hello, who are you?"},
        {"role":"assistant", "content":"<think>User is asking about my identity.</think>I am a helpful assistant."},
        {"role":"user", "content":"What is the weather in New York?"}
    ]
    ```
    
    ```python
    <|role_start|>system<|role_end|>
    You are a helpful assistant.
    <|role_start|>user<|role_end|>
    Hello, who are you?
    <|role_start|>assistant<|role_end|>
    I am a helpful assistant.
    <|role_start|>user<|role_end|>
    What is the weather in New York?
    <|role_start|>assistant<|role_end|>
    <think>
    
    ```
    
- Multi Turn - system, user, assistant (add_generation_prompt=False)
    
    ```python
    chat = [
        {"role":"system", "content":"You are a helpful assistant."},
        {"role":"user", "content":"Hello, who are you?"},
        {"role":"assistant", "content":"<think>User is asking about my identity.</think>I am a helpful assistant."},
        {"role":"user", "content":"What is the weather in New York?"},
        {"role":"assistant", "content":"<think>User is asking about the weather in New York.</think>The weather in New York is sunny."}
    ]
    ```
    
    ```python
    <|role_start|>system<|role_end|>
    You are a helpful assistant.
    <|role_start|>user<|role_end|>
    Hello, who are you?
    <|role_start|>assistant<|role_end|>
    I am a helpful assistant.</content>
    <|role_start|>user<|role_end|>
    What is the weather in New York?
    <|role_start|>assistant<|role_end|>
    <think>
    User is asking about the weather in New York.
    </think>
    
    The weather in New York is sunny.<|END|>
    
    ```
    
- Tool call
    
    ```python
    chat = [
        {"role":"system", "content":"You are a helpful assistant."},
        {"role":"user", "content":"Hello, let's tool call."},
        {
          "role": "assistant",
          "content": "",
          "tool_calls": [
            {
              "id": "call_1ad776a1218d4f88ba3f",
              "type": "function",
              "function": {
                "name": "reddit-content-fetcher-fetch_reddit_hot_threads",
                "arguments": "{\"subreddit\": \"programming\", \"limit\": 15}"
              }
            },
            {
              "id": "",
              "type": "function",
              "function": {
                "name": "rasdasdreads",
                "arguments": "{\"subreddit\": \"programming\", \"limit\": 15}"
              }
            },
          ]
        },
        {
          "role": "tool",
          "tool_call_id": "call_1ad776a1218d4f88ba3f",
          "name": "reddit-content-fetcher-fetch_reddit_hot_threads_1",
          "content": "Title: The enshittification of tech jobs\nScore: 237\nComments: 77\nAuthor: JumbleGuide\nType: link\nContent: https://www.reddit.com/r/programming/comments/1mjygiu/the_enshittification_of_tech_jobs/\nLink: https://reddit.comhttps://www.reddit.com/r/programming/comments/1mjygiu/the_enshittification_of_tech_jobs/\n---"
        },
        {
          "role": "tool",
          "tool_call_id": "call_1ad776a1218d4f88ba3f",
        
          "name": "reddit-content-fetcher-fetch_reddit_hot_threads_2",
          "content": "Title: The enshittification of tech jobs\nScore: 237\nComments: 77\nAuthor: JumbleGuide\nType: link\nContent: https://www.reddit.com/r/programming/comments/1mjygiu/the_enshittification_of_tech_jobs/\nLink: https://reddit.comhttps://www.reddit.com/r/programming/comments/1mjygiu/the_enshittification_of_tech_jobs/\n---"
        },
    ]
    ```
    
    ```python
    "tools": [
        {
          "type": "function",
          "function": {
            "name": "reddit-content-fetcher-fetch_reddit_hot_threads",
            "description": "\n    Fetch hot threads from a subreddit\n    \n    Args:\n        subreddit: Name of the subreddit\n        limit: Number of posts to fetch (default: 10)\n        \n    Returns:\n        Human readable string containing list of post information\n    ",
            "parameters": {
              "additionalProperties": False,
              "properties": {
                "subreddit": {
                  "title": "Subreddit",
                  "type": "string"
                },
                "limit": {
                  "default": 10,
                  "title": "Limit",
                  "type": "integer"
                }
              },
              "required": [
                "subreddit"
              ],
              "type": "object"
            }
          }
        },
        {
          "type": "function",
          "function": {
            "name": "reddit-content-fetcher-fetch_reddit_post_content",
            "description": "\n    Fetch detailed content of a specific post\n    \n    Args:\n        post_id: Reddit post ID\n        comment_limit: Number of top level comments to fetch\n        comment_depth: Maximum depth of comment tree to traverse\n\n    Returns:\n        Human readable string containing post content and comments tree\n    ",
            "parameters": {
              "additionalProperties": False,
              "properties": {
                "post_id": {
                  "title": "Post Id",
                  "type": "string"
                },
                "comment_limit": {
                  "default": 20,
                  "title": "Comment Limit",
                  "type": "integer"
                },
                "comment_depth": {
                  "default": 3,
                  "title": "Comment Depth",
                  "type": "integer"
                }
              },
              "required": [
                "post_id"
              ],
              "type": "object"
            }
          }
        }
      ]
    ```
    
    ```python
    <|role_start|>system<|role_end|>
    You are a helpful assistant.
    
    ## Tools ##
    
    You can call one or more available functions to help with the user’s query.
    
    Each function is wrapped in <tool_start> and <tool_end> tags inside the <tools></tools> block:
    <tools>
    <tool_start>
    {"type": "function", "function": {"name": "reddit-content-fetcher-fetch_reddit_hot_threads", "description": "\n    Fetch hot threads from a subreddit\n    \n    Args:\n        subreddit: Name of the subreddit\n        limit: Number of posts to fetch (default: 10)\n        \n    Returns:\n        Human readable string containing list of post information\n    ", "parameters": {"additionalProperties": false, "properties": {"subreddit": {"title": "Subreddit", "type": "string"}, "limit": {"default": 10, "title": "Limit", "type": "integer"}}, "required": ["subreddit"], "type": "object"}}}
    <tool_end>
    <tool_start>
    {"type": "function", "function": {"name": "reddit-content-fetcher-fetch_reddit_post_content", "description": "\n    Fetch detailed content of a specific post\n    \n    Args:\n        post_id: Reddit post ID\n        comment_limit: Number of top level comments to fetch\n        comment_depth: Maximum depth of comment tree to traverse\n\n    Returns:\n        Human readable string containing post content and comments tree\n    ", "parameters": {"additionalProperties": false, "properties": {"post_id": {"title": "Post Id", "type": "string"}, "comment_limit": {"default": 20, "title": "Comment Limit", "type": "integer"}, "comment_depth": {"default": 3, "title": "Comment Depth", "type": "integer"}}, "required": ["post_id"], "type": "object"}}}
    <tool_end>
    </tools>
    
    Each time a function is called, return a JSON object specifying the function name and arguments, enclosed within <tool_call></tool_call> tags:
    <tool_call>
    {"name": <function-name>, "arguments": <args-json-object>}
    </tool_call></content>
    <role>user</role>
    <content>Hello, let's tool call.</content>
    <role>assistant</role>
    <content><tool_call>
    {"name": "reddit-content-fetcher-fetch_reddit_hot_threads", "arguments": {"subreddit": "programming", "limit": 15}}
    </tool_call>
    <tool_call>
    {"name": "rasdasdreads", "arguments": {"subreddit": "programming", "limit": 15}}
    </tool_call>
    <|role_start|>user<|role_end|>
    <response>
    Title: The enshittification of tech jobs
    Score: 237
    Comments: 77
    Author: JumbleGuide
    Type: link
    Content: https://www.reddit.com/r/programming/comments/1mjygiu/the_enshittification_of_tech_jobs/
    Link: https://reddit.comhttps://www.reddit.com/r/programming/comments/1mjygiu/the_enshittification_of_tech_jobs/
    ---
    </tool_response>
    <tool_response>
    Title: The enshittification of tech jobs
    Score: 237
    Comments: 77
    Author: JumbleGuide
    Type: link
    Content: https://www.reddit.com/r/programming/comments/1mjygiu/the_enshittification_of_tech_jobs/
    Link: https://reddit.comhttps://www.reddit.com/r/programming/comments/1mjygiu/the_enshittification_of_tech_jobs/
    ---
    </tool_response>
    <|role_start|>assistant<|role_end|>
    <think>
    
    </think>
    
    ```