import axios from 'axios';

const api = axios.create({ baseURL: '/api/v1' });

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
