/**
 * WebSocket client for real-time analysis streaming.
 * Falls back to REST polling if WebSocket is unavailable.
 */

type EventHandler = (event: AnalysisEvent) => void;

interface AnalysisEvent {
  type: 'status_update' | 'alpha_prediction' | 'agent_view' | 'agent_debate' | 'memo' | 'error';
  data: Record<string, unknown>;
}

export class AnalysisWebSocket {
  private ws: WebSocket | null = null;
  private lastEventIndex = 0;
  private handlers: EventHandler[] = [];
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private analysisId: string | null = null;

  onEvent(handler: EventHandler) {
    this.handlers.push(handler);
    return () => {
      this.handlers = this.handlers.filter(h => h !== handler);
    };
  }

  connect(analysisId: string) {
    this.analysisId = analysisId;
    this.reconnectAttempts = 0;

    const backendUrl = import.meta.env.VITE_API_URL || '';
    let url: string;
    if (backendUrl) {
      // External backend: convert https://foo.com/api/v1 → wss://foo.com/api/v1/ws/...
      const wsBase = backendUrl.replace(/^http/, 'ws');
      url = `${wsBase}/ws/analysis/${analysisId}`;
    } else {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.host;
      url = `${protocol}//${host}/api/v1/ws/analysis/${analysisId}`;
    }

    try {
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        console.log(`[WS] Connected to analysis ${analysisId}`);
      };

      this.ws.onmessage = (e) => {
        try {
          const event: AnalysisEvent = JSON.parse(e.data);
          this.lastEventIndex++;
          this.handlers.forEach(h => h(event));
        } catch (err) {
          console.warn('[WS] Failed to parse message:', err);
        }
      };

      this.ws.onclose = (e) => {
        console.log(`[WS] Disconnected (code=${e.code})`);
        if (this.reconnectAttempts < this.maxReconnectAttempts && e.code !== 1000) {
          const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 10000);
          const jitter = Math.random() * 1000;
          setTimeout(() => this.reconnect(), delay + jitter);
          this.reconnectAttempts++;
        }
      };

      this.ws.onerror = (err) => {
        console.error('[WS] Error:', err);
      };
    } catch {
      console.warn('[WS] WebSocket not available, falling back to polling');
    }
  }

  private reconnect() {
    if (this.analysisId) {
      console.log(`[WS] Reconnecting (attempt ${this.reconnectAttempts})...`);
      this.connect(this.analysisId);
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    this.analysisId = null;
    this.handlers = [];
  }

  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

// Singleton instance
export const analysisWs = new AnalysisWebSocket();
