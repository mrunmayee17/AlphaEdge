import axios from 'axios';

const BACKEND_URL = import.meta.env.VITE_API_URL || '/api/v1';
const api = axios.create({ baseURL: BACKEND_URL });

export async function startAnalysis(ticker: string) {
  const { data } = await api.post('/analysis', { ticker });
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
