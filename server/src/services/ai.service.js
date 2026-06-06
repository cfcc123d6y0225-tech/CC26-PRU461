import axios from "axios";

const AI_SERVICE_URL = process.env.AI_SERVICE_URL;

export const predictMajor = async (featureVector) => {
  try {
    const response = await axios.post(`${AI_SERVICE_URL}/predict`, {
      features: featureVector,
    });

    return response.data;
  } catch (error) {
    console.error("AI Service Error:", error.message);
    throw error;
  }
};
