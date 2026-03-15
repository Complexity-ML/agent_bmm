FROM python:3.12-slim

WORKDIR /app

# Install minimal deps first (cache-friendly)
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy source
COPY agent_bmm/ agent_bmm/

EXPOSE 8765

ENTRYPOINT ["agent-bmm"]
CMD ["serve"]
