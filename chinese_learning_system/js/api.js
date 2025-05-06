// api.js - Handles all API calls to the server

const api = {
  // Base URL for API calls
  baseURL: "/api",

  // Helper method for making GET requests
  async get(endpoint) {
    try {
      const response = await fetch(`${this.baseURL}${endpoint}`);
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error(`Error fetching ${endpoint}:`, error);
      throw error;
    }
  },

  // Helper method for making POST requests
  async post(endpoint, data) {
    try {
      const response = await fetch(`${this.baseURL}${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`Error posting to ${endpoint}:`, error);
      throw error;
    }
  },

  // Helper method for uploading files
  async upload(endpoint, formData) {
    try {
      const response = await fetch(`${this.baseURL}${endpoint}`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`Error uploading to ${endpoint}:`, error);
      throw error;
    }
  },

  // Student profile methods
  async getStudentProfile() {
    return this.get("/student/profile");
  },

  async saveStudentProfile(profile) {
    return this.post("/student/profile", profile);
  },

  // Learning style survey methods
  async generateLearningStyleSurvey() {
    return this.get("/survey/learning-style");
  },

  async submitLearningStyleSurvey(answers) {
    return this.post("/survey/learning-style", { answers });
  },

  // Dashboard data
  async getDashboardData() {
    return this.get("/dashboard");
  },

  // Document upload methods
  async uploadDocuments(formData) {
    return this.upload("/documents/upload", formData);
  },

  // Test methods
  async getTestAvailability() {
    return this.get("/tests/availability");
  },

  async generatePretest() {
    return this.get("/tests/pretest");
  },

  async generatePosttest() {
    return this.get("/tests/posttest");
  },

  async submitTestAnswers(testType, answers) {
    return this.post(`/tests/${testType}/submit`, { answers });
  },

  // Learning modules methods
  async getLearningModules() {
    return this.get("/modules");
  },

  async getModuleContent(moduleId) {
    return this.get(`/modules/${moduleId}/content`);
  },

  // Discussion methods
  async getDiscussionTopics() {
    return this.get("/discussion/topics");
  },

  async getDiscussionTopic(topicId) {
    return this.get(`/discussion/topics/${topicId}`);
  },

  async sendChatMessage(topicId, message) {
    return this.post(`/discussion/topics/${topicId}/message`, { message });
  },

  // Learning logs methods
  async getLearningLogs() {
    return this.get("/logs");
  },

  async getLearningLog(logId) {
    return this.get(`/logs/${logId}`);
  },

  async saveLearningLog(log) {
    return this.post("/logs", log);
  },

  async analyzeLearningLog(logId) {
    return this.get(`/logs/${logId}/analyze`);
  },
};
