import axios from 'axios';

const API_URL = 'http://127.0.0.1:5000/api/emissions';

export const getAllEmissions = async () => {
    const response = await axios.get(API_URL);
    return response.data;
};

export const getStateEmissions = async (state: string) => {
    const response = await axios.get(`${API_URL}?state=${state}`);
    return response.data;
};