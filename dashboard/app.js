const setText = (field, value) => {
  document.querySelector(`[data-field="${field}"]`).textContent = value ?? "—";
};

fetch("/api/status")
  .then((response) => response.json())
  .then((payload) => {
    setText("mode", payload.mode);
    setText("status", payload.status);
    const market = payload.market_data;
    setText("symbol", market?.symbol);
    setText("timestamp_utc", market?.timestamp_utc);
    setText("close", market?.close);
    setText("row_count", market?.row_count);
  })
  .catch(() => setText("status", "data_unavailable"));
