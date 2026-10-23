# Chạy TradingAgents bằng Codex

Provider `codex` gọi CLI chính thức qua `codex exec`, dùng phiên đăng nhập
ChatGPT của bạn. Không cần OpenAI API key cho phần LLM.

## Cài và chạy

Trong thư mục repo:

```bash
uv sync --extra dev
npm install -g @openai/codex
codex login
codex login status
cp .env.codex.example .env
uv run tradingagents
```

Nếu đã có `.env`, ghép cấu hình từ `.env.codex.example` vào thay vì ghi đè.
CLI được đối chiếu với Codex **0.163.0**. Bản cũ thiếu
`--ignore-user-config` cần cập nhật.

Chạy BTC chỉ với analyst thị trường, thay ngày bằng ngày cần phân tích:

```bash
uv run tradingagents --ticker BTC-USD --date YYYY-MM-DD \
  --analysts market --save --no-show
```

Đây là chạy phân tích và lưu báo cáo, không phải lệnh đặt giao dịch.
Dữ liệu thị trường vẫn do các data vendor của TradingAgents cung cấp.
Chuyển LLM sang Codex không thay thế key của FRED, Alpha Vantage hay nguồn
dữ liệu khác nếu bạn chọn chúng. Cấu hình mặc định giá dùng Yahoo Finance.

## Cấu hình

| Biến | Ý nghĩa |
|---|---|
| `TRADINGAGENTS_LLM_PROVIDER=codex` | Chọn bridge Codex |
| `TRADINGAGENTS_QUICK_THINK_LLM=default` | Model mặc định tích hợp trong CLI cho analyst |
| `TRADINGAGENTS_DEEP_THINK_LLM=default` | Model mặc định tích hợp trong CLI cho manager |
| `TRADINGAGENTS_CODEX_COMMAND=codex` | Tên executable hoặc đường dẫn tuyệt đối, không phải chuỗi shell |
| `TRADINGAGENTS_CODEX_TIMEOUT=1800` | Timeout giây cho mỗi lượt model |
| `TRADINGAGENTS_OPENAI_REASONING_EFFORT=low` | Mức reasoning; phải được model hỗ trợ |

Có thể thay `default` bằng model ID Codex mà tài khoản của bạn truy cập được.
Hai biến `*_THINK_PROVIDER=codex` trong mẫu bảo đảm cả hai tầng dùng Codex.
Xóa cấu hình endpoint (`*_BACKEND_URL`), temperature, max_tokens và
LLM retry riêng từ cấu hình API cũ. Bridge báo lỗi với tham số chưa hỗ trợ,
không âm thầm bỏ qua. Không tự retry ở cấp TradingAgents; CLI có thể có
cơ chế thử lại kết nối của chính nó.

Trong Python:

```python
from tradingagents.default_config import build_default_config
from tradingagents.graph.trading_graph import TradingAgentsGraph

config = build_default_config()
config.update(
    llm_provider="codex",
    quick_think_provider="codex", deep_think_provider="codex",
    quick_think_llm="default", deep_think_llm="default",
    backend_url=None,
    quick_think_backend_url=None, deep_think_backend_url=None,
    temperature=None, max_tokens=None, llm_max_retries=None,
)
graph = TradingAgentsGraph(selected_analysts=["market"], config=config)
# state, decision = graph.propagate("BTC-USD", "YYYY-MM-DD")
```

## Cách hoạt động và giới hạn

- Mỗi lượt là một session ephemeral trong thư mục tạm, sandbox read-only.
  CLI bỏ qua config cá nhân để tránh nạp MCP, hooks hoặc endpoint tùy chỉnh;
  vẫn dùng cơ chế đăng nhập chính thức của CLI. Shell, apps và hooks bị tắt.
- Hội thoại, tool schema và tool result được gửi trong prompt. Codex trả JSON
  gồm nội dung hoặc yêu cầu tool; **LangGraph** thực thi các tool ứng dụng.
- Hỗ trợ `bind_tools`, `with_structured_output` và kiểm tra kiểu dữ liệu.
  Chưa hỗ trợ ảnh, stop sequences hoặc streaming theo token. Không giả số
  token/cost khi CLI không cung cấp chúng qua bridge.
- Không đọc hay sao chép token đăng nhập. `OPENAI_API_KEY` và `CODEX_API_KEY`
  bị bỏ khỏi môi trường process con; bắt buộc login ChatGPT, không fallback API.
- Lượt phân tích nhiều agent có thể chậm và tiêu hao hạn mức gói ChatGPT.
  Khi lỗi, kiểm tra `codex login status`, model ID, hạn mức và timeout.
- Đây là bridge qua prompt/JSON, không phải API tool calling nguyên bản.
  Test giao thức không chứng minh chất lượng hay độ chính xác phân tích tài chính.

Tài liệu chính thức: [Codex CLI](https://developers.openai.com/codex/cli/),
[non-interactive mode](https://developers.openai.com/codex/noninteractive),
[authentication](https://developers.openai.com/codex/auth).

## Kiểm tra offline

```bash
uv run pytest tests/test_codex_provider.py -q
uv run pytest -q
```

Test bridge sử dụng subprocess giả lập, không gửi dữ liệu và không tiêu hao
hạn mức model. Bài test live phải chạy riêng với tài khoản đã đăng nhập.

Kiểm tra gần nhất: **18 tests của bridge Codex đã qua**. Các tests dùng
subprocess giả lập để kiểm tra giao thức, tham số, tool calling và xử lý lỗi.

**Trạng thái live: chưa xác nhận chạy hoàn tất end-to-end.** Trong môi trường
thử nghiệm, CLI báo đã đăng nhập nhưng lời gọi model không hoàn tất. Một bài
thử tối giản cũng không trả lời. Không coi `codex login status` hoặc tests giả
lập là bằng chứng inference thành công.

Mẫu cấu hình đặt timeout **1.800 giây cho mỗi lần gọi model**, không phải cho
cả lượt phân tích. Việc tăng timeout chưa được chứng minh khắc phục lỗi.
Trước khi chạy bài phân tích dài, kiểm tra một prompt thực tế bằng Codex CLI
trên máy đã đăng nhập của bạn. Không tự chuyển sang API key khi Codex lỗi.
