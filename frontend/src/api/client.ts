import axios from 'axios';
import type { ForecastModel } from '../types';

const BACKEND_URL = import.meta.env.VITE_API_URL || '/api/v1';
const api = axios.create({
  baseURL: BACKEND_URL,
  timeout: 30000,
});

api.interceptors.response.use(
  response => response,
  (error) => {
    if (error.response) {
      const status = error.response.status;
      const detail = error.response.data?.detail;
      const message = detail ? `${status}: ${detail}` : `HTTP ${status}`;
      return Promise.reject(new Error(message));
    }
    if (error.code === 'ECONNABORTED') {
      return Promise.reject(new Error('Request timed out'));
    }
    return Promise.reject(new Error('Network request failed'));
  },
);

export async function startAnalysis(ticker: string, forecastModel: ForecastModel) {
  const { data } = await api.post('/analysis', { ticker, forecast_model: forecastModel });
  return data;
}

export async function getAnalysisStatus(analysisId: string) {
  const { data } = await api.get(`/analysis/${analysisId}`);
  return data;
}

export async function getMemo(analysisId: string) {
  const { data } = await api.get(`/analysis/${analysisId}/memo`);
  return data;
}
