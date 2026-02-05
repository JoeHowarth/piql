# piql-http

HTTP server for piql queries with automatic file watching.

## Usage

```bash
# Watch a directory (loads all parquet, csv, ipc, arrow files)
piql-http ./data/

# Watch specific files
piql-http entities.parquet locations.csv

# Custom port/host
piql-http -p 8080 --host 0.0.0.0 ./data/
```

Files are mapped to dataframe names by stem: `data/entities.parquet` → `"entities"`

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Execute piql query (body: query string, response: Arrow IPC) |
| `/ask` | POST | Natural language to piql (body: question, response: query in header, optionally Arrow IPC in body) |
| `/dataframes` | GET | List available dataframe names |
| `/openapi.json` | GET | OpenAPI 3.1 spec |
| `/swagger-ui` | GET | Interactive API docs |

## Examples

```bash
# List dataframes
curl http://localhost:3000/dataframes
# {"names":["entities","locations"]}

# Run a query
curl -X POST http://localhost:3000/query \
  -H 'Content-Type: text/plain' \
  -d 'entities.filter(pl.col("gold") > 100)'
# Returns Arrow IPC bytes

# Natural language query (generate only)
curl -i -X POST http://localhost:3000/ask \
  -H 'Content-Type: text/plain' \
  -d 'Show me the top 10 entities by gold'
# Returns: X-Piql-Query header with generated query, empty body

# Natural language query (generate and execute)
curl -X POST 'http://localhost:3000/ask?execute=true' \
  -H 'Content-Type: text/plain' \
  -d 'How many entities of each type?'
# Returns: X-Piql-Query header with query, Arrow IPC bytes in body
```

## Natural Language Queries

The `/ask` endpoint converts natural language questions to piql queries using an LLM:

- If `OPENROUTER_API_KEY` env var is set: Uses OpenRouter API (claude-sonnet-4)
- Otherwise: Falls back to `claude -p` CLI

Response format:
- `X-Piql-Query` header: Always contains the generated query
- Body: Empty if `execute=false` (default), Arrow IPC bytes if `execute=true`

## TypeScript Client Generation

```bash
npx @hey-api/openapi-ts -i http://localhost:3000/openapi.json -o ./client
```

## File Watching

The server watches for changes to data files and automatically reloads them. Supported formats:
- `.parquet`
- `.csv`
- `.ipc` / `.arrow`
