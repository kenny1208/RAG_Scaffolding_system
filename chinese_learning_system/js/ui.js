// ui.js - UI helpers and components

const ui = {
  // Show notification to the user
  showNotification(message, type = "info") {
    // Create notification element if it doesn't exist
    let notificationContainer = document.querySelector(
      ".notification-container"
    );

    if (!notificationContainer) {
      notificationContainer = document.createElement("div");
      notificationContainer.className = "notification-container";
      document.body.appendChild(notificationContainer);
    }

    // Create notification
    const notification = document.createElement("div");
    notification.className = `notification ${type}`;

    const messageElement = document.createElement("span");
    messageElement.textContent = message;

    const closeButton = document.createElement("button");
    closeButton.textContent = "×";
    closeButton.addEventListener("click", () => {
      notification.classList.add("hiding");
      setTimeout(() => {
        notification.remove();
      }, 300);
    });

    notification.appendChild(messageElement);
    notification.appendChild(closeButton);
    notificationContainer.appendChild(notification);

    // Auto-hide notification after 5 seconds
    setTimeout(() => {
      notification.classList.add("hiding");
      setTimeout(() => {
        notification.remove();
      }, 300);
    }, 5000);
  },

  // Show confirmation dialog
  showConfirmation(message, confirmCallback, cancelCallback = null) {
    // Create modal container if it doesn't exist
    let modalContainer = document.querySelector(".modal-container");

    if (!modalContainer) {
      modalContainer = document.createElement("div");
      modalContainer.className = "modal-container";
      document.body.appendChild(modalContainer);
    }

    // Create confirmation modal
    const modal = document.createElement("div");
    modal.className = "modal confirmation-modal";

    const modalContent = document.createElement("div");
    modalContent.className = "modal-content";

    const messageElement = document.createElement("p");
    messageElement.textContent = message;

    const buttonContainer = document.createElement("div");
    buttonContainer.className = "button-container";

    const confirmButton = document.createElement("button");
    confirmButton.className = "primary-btn";
    confirmButton.textContent = "確認";
    confirmButton.addEventListener("click", () => {
      modal.remove();
      if (confirmCallback) confirmCallback();
    });

    const cancelButton = document.createElement("button");
    cancelButton.className = "secondary-btn";
    cancelButton.textContent = "取消";
    cancelButton.addEventListener("click", () => {
      modal.remove();
      if (cancelCallback) cancelCallback();
    });

    buttonContainer.appendChild(confirmButton);
    buttonContainer.appendChild(cancelButton);

    modalContent.appendChild(messageElement);
    modalContent.appendChild(buttonContainer);
    modal.appendChild(modalContent);
    modalContainer.appendChild(modal);
  },

  // Show loading indicator
  showLoading(message = "載入中...") {
    // Create loading container if it doesn't exist
    let loadingContainer = document.querySelector(".loading-container");

    if (!loadingContainer) {
      loadingContainer = document.createElement("div");
      loadingContainer.className = "loading-container";
      document.body.appendChild(loadingContainer);
    }

    // Clear existing content
    loadingContainer.innerHTML = "";

    // Create loading indicator
    const loadingElement = document.createElement("div");
    loadingElement.className = "loading-indicator";

    const spinner = document.createElement("div");
    spinner.className = "spinner";

    const messageElement = document.createElement("p");
    messageElement.textContent = message;

    loadingElement.appendChild(spinner);
    loadingElement.appendChild(messageElement);
    loadingContainer.appendChild(loadingElement);

    // Show loading container
    loadingContainer.classList.add("show");
  },

  // Hide loading indicator
  hideLoading() {
    const loadingContainer = document.querySelector(".loading-container");
    if (loadingContainer) {
      loadingContainer.classList.remove("show");
    }
  },

  // Format dates
  formatDate(dateString) {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat("zh-TW", {
      year: "numeric",
      month: "long",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(date);
  },

  // Update feature availability based on user progress
  updateFeatureAvailability(profile) {
    // Update tests availability
    if (profile && profile.learning_style) {
      document.getElementById("start-pretest").disabled = false;
    } else {
      document.getElementById("start-pretest").disabled = true;
    }

    // Update posttest availability
    if (
      profile &&
      profile.learning_history &&
      profile.learning_history.some((h) => h.activity_type === "前測")
    ) {
      document.getElementById("start-posttest").disabled = false;
    } else {
      document.getElementById("start-posttest").disabled = true;
    }
  },

  // Load learning logs
  loadLearningLogs() {
    api
      .getLearningLogs()
      .then((logs) => {
        const logsList = document.getElementById("logs-list");
        logsList.innerHTML = "";

        if (logs && logs.length > 0) {
          logs.forEach((log) => {
            const logItem = document.createElement("li");
            logItem.setAttribute("data-log-id", log.id);
            logItem.textContent = `${log.topic} - ${this.formatDate(
              log.timestamp
            )}`;

            logItem.addEventListener("click", () => {
              // Update selected log
              document.querySelectorAll("#logs-list li").forEach((item) => {
                item.classList.remove("active");
              });
              logItem.classList.add("active");

              // Load log content
              this.loadLogContent(log.id);
            });

            logsList.appendChild(logItem);
          });
        } else {
          const emptyItem = document.createElement("li");
          emptyItem.className = "empty-state";
          emptyItem.textContent = "尚無學習日誌";
          logsList.appendChild(emptyItem);
        }
      })
      .catch((error) => {
        console.error("Error loading learning logs:", error);
        this.showNotification("載入學習日誌時發生錯誤", "error");
      });
  },

  // Load log content
  loadLogContent(logId) {
    api
      .getLearningLog(logId)
      .then((log) => {
        document.getElementById("log-editor").classList.add("hidden");

        const logDisplay = document.getElementById("log-display");
        logDisplay.classList.remove("hidden");
        logDisplay.innerHTML = "";

        // Create log header
        const header = document.createElement("div");
        header.className = "log-display-header";

        const title = document.createElement("h3");
        title.textContent = log.topic;

        const date = document.createElement("span");
        date.className = "log-date";
        date.textContent = this.formatDate(log.timestamp);

        header.appendChild(title);
        header.appendChild(date);

        // Create log content
        const content = document.createElement("div");
        content.className = "log-section";

        const contentTitle = document.createElement("h4");
        contentTitle.textContent = "學習內容";

        const contentText = document.createElement("p");
        contentText.textContent = log.content;

        content.appendChild(contentTitle);
        content.appendChild(contentText);

        // Create reflections
        const reflections = document.createElement("div");
        reflections.className = "log-section";

        const reflectionsTitle = document.createElement("h4");
        reflectionsTitle.textContent = "反思";

        reflections.appendChild(reflectionsTitle);

        if (log.reflections && log.reflections.length > 0) {
          const reflectionsList = document.createElement("ul");
          log.reflections.forEach((reflection) => {
            const reflectionItem = document.createElement("li");
            reflectionItem.textContent = reflection;
            reflectionsList.appendChild(reflectionItem);
          });
          reflections.appendChild(reflectionsList);
        } else {
          const noReflections = document.createElement("p");
          noReflections.textContent = "尚無反思記錄";
          reflections.appendChild(noReflections);
        }

        // Create questions
        const questions = document.createElement("div");
        questions.className = "log-section";

        const questionsTitle = document.createElement("h4");
        questionsTitle.textContent = "問題";

        questions.appendChild(questionsTitle);

        if (log.questions && log.questions.length > 0) {
          const questionsList = document.createElement("ul");
          log.questions.forEach((question) => {
            const questionItem = document.createElement("li");
            questionItem.textContent = question;
            questionsList.appendChild(questionItem);
          });
          questions.appendChild(questionsList);
        } else {
          const noQuestions = document.createElement("p");
          noQuestions.textContent = "尚無問題記錄";
          questions.appendChild(noQuestions);
        }

        // Create next steps
        const nextSteps = document.createElement("div");
        nextSteps.className = "log-section";

        const nextStepsTitle = document.createElement("h4");
        nextStepsTitle.textContent = "後續步驟";

        nextSteps.appendChild(nextStepsTitle);

        if (log.next_steps && log.next_steps.length > 0) {
          const nextStepsList = document.createElement("ul");
          log.next_steps.forEach((step) => {
            const stepItem = document.createElement("li");
            stepItem.textContent = step;
            nextStepsList.appendChild(stepItem);
          });
          nextSteps.appendChild(nextStepsList);
        } else {
          const noSteps = document.createElement("p");
          noSteps.textContent = "尚無後續步驟記錄";
          nextSteps.appendChild(noSteps);
        }

        // Add analyze button
        const analyzeButton = document.createElement("button");
        analyzeButton.className = "primary-btn";
        analyzeButton.textContent = "分析日誌";
        analyzeButton.addEventListener("click", () => {
          this.analyzeLog(logId);
        });

        // Add edit button
        const editButton = document.createElement("button");
        editButton.className = "secondary-btn";
        editButton.textContent = "編輯日誌";
        editButton.addEventListener("click", () => {
          this.editLog(log);
        });

        const buttonContainer = document.createElement("div");
        buttonContainer.className = "button-container";
        buttonContainer.appendChild(analyzeButton);
        buttonContainer.appendChild(editButton);

        // Append all sections
        logDisplay.appendChild(header);
        logDisplay.appendChild(content);
        logDisplay.appendChild(reflections);
        logDisplay.appendChild(questions);
        logDisplay.appendChild(nextSteps);
        logDisplay.appendChild(buttonContainer);
      })
      .catch((error) => {
        console.error("Error loading log content:", error);
        this.showNotification("載入日誌內容時發生錯誤", "error");
      });
  },

  // Analyze log
  analyzeLog(logId) {
    this.showLoading("分析日誌中...");

    api
      .analyzeLearningLog(logId)
      .then((analysis) => {
        this.hideLoading();

        // Create analysis modal
        let modalContainer = document.querySelector(".modal-container");

        if (!modalContainer) {
          modalContainer = document.createElement("div");
          modalContainer.className = "modal-container";
          document.body.appendChild(modalContainer);
        }

        const modal = document.createElement("div");
        modal.className = "modal analysis-modal";

        const modalContent = document.createElement("div");
        modalContent.className = "modal-content";

        const modalHeader = document.createElement("div");
        modalHeader.className = "modal-header";

        const title = document.createElement("h3");
        title.textContent = "日誌分析結果";

        const closeButton = document.createElement("span");
        closeButton.className = "close-modal";
        closeButton.textContent = "×";
        closeButton.addEventListener("click", () => {
          modal.remove();
        });

        modalHeader.appendChild(title);
        modalHeader.appendChild(closeButton);

        const modalBody = document.createElement("div");
        modalBody.className = "modal-body";

        // Create analysis sections
        const understandingLevel = document.createElement("div");
        understandingLevel.className = "analysis-section";

        const understandingTitle = document.createElement("h4");
        understandingTitle.textContent = "理解程度";

        const understandingText = document.createElement("p");
        understandingText.textContent = analysis.understanding_level;

        understandingLevel.appendChild(understandingTitle);
        understandingLevel.appendChild(understandingText);

        // Create strengths section
        const strengths = document.createElement("div");
        strengths.className = "analysis-section";

        const strengthsTitle = document.createElement("h4");
        strengthsTitle.textContent = "優點";

        strengths.appendChild(strengthsTitle);

        if (analysis.strengths && analysis.strengths.length > 0) {
          const strengthsList = document.createElement("ul");
          analysis.strengths.forEach((strength) => {
            const strengthItem = document.createElement("li");
            strengthItem.textContent = strength;
            strengthsList.appendChild(strengthItem);
          });
          strengths.appendChild(strengthsList);
        } else {
          const noStrengths = document.createElement("p");
          noStrengths.textContent = "未識別到特定優點";
          strengths.appendChild(noStrengths);
        }

        // Create areas for improvement section
        const improvements = document.createElement("div");
        improvements.className = "analysis-section";

        const improvementsTitle = document.createElement("h4");
        improvementsTitle.textContent = "需要改進的領域";

        improvements.appendChild(improvementsTitle);

        if (
          analysis.areas_for_improvement &&
          analysis.areas_for_improvement.length > 0
        ) {
          const improvementsList = document.createElement("ul");
          analysis.areas_for_improvement.forEach((area) => {
            const areaItem = document.createElement("li");
            areaItem.textContent = area;
            improvementsList.appendChild(areaItem);
          });
          improvements.appendChild(improvementsList);
        } else {
          const noImprovements = document.createElement("p");
          noImprovements.textContent = "未識別到需要改進的領域";
          improvements.appendChild(noImprovements);
        }

        // Create emotional response section
        const emotional = document.createElement("div");
        emotional.className = "analysis-section";

        const emotionalTitle = document.createElement("h4");
        emotionalTitle.textContent = "情感回應";

        const emotionalText = document.createElement("p");
        emotionalText.textContent = analysis.emotional_response;

        emotional.appendChild(emotionalTitle);
        emotional.appendChild(emotionalText);

        // Create recommended next steps section
        const nextSteps = document.createElement("div");
        nextSteps.className = "analysis-section";

        const nextStepsTitle = document.createElement("h4");
        nextStepsTitle.textContent = "建議後續步驟";

        nextSteps.appendChild(nextStepsTitle);

        if (
          analysis.recommended_next_steps &&
          analysis.recommended_next_steps.length > 0
        ) {
          const nextStepsList = document.createElement("ul");
          analysis.recommended_next_steps.forEach((step) => {
            const stepItem = document.createElement("li");
            stepItem.textContent = step;
            nextStepsList.appendChild(stepItem);
          });
          nextSteps.appendChild(nextStepsList);
        } else {
          const noSteps = document.createElement("p");
          noSteps.textContent = "未提供建議後續步驟";
          nextSteps.appendChild(noSteps);
        }

        // Append all sections
        modalBody.appendChild(understandingLevel);
        modalBody.appendChild(strengths);
        modalBody.appendChild(improvements);
        modalBody.appendChild(emotional);
        modalBody.appendChild(nextSteps);

        // Create modal footer
        const modalFooter = document.createElement("div");
        modalFooter.className = "modal-footer";

        const closeModalButton = document.createElement("button");
        closeModalButton.className = "primary-btn";
        closeModalButton.textContent = "關閉";
        closeModalButton.addEventListener("click", () => {
          modal.remove();
        });

        modalFooter.appendChild(closeModalButton);

        // Append all modal parts
        modalContent.appendChild(modalHeader);
        modalContent.appendChild(modalBody);
        modalContent.appendChild(modalFooter);
        modal.appendChild(modalContent);
        modalContainer.appendChild(modal);
      })
      .catch((error) => {
        this.hideLoading();
        console.error("Error analyzing log:", error);
        this.showNotification("分析日誌時發生錯誤", "error");
      });
  },

  // Edit log
  editLog(log) {
    document.getElementById("log-display").classList.add("hidden");

    const logEditor = document.getElementById("log-editor");
    logEditor.classList.remove("hidden");

    document.getElementById("log-topic").value = log.topic;
    document.getElementById("log-content").value = log.content;

    // Set reflections if available
    let reflections = "";
    if (log.reflections && log.reflections.length > 0) {
      reflections = log.reflections.join("\n");
    }
    document.getElementById("log-reflection").value = reflections;

    // Change the save button to update
    const saveButton = document.getElementById("save-log");
    saveButton.textContent = "更新日誌";
    saveButton.setAttribute("data-log-id", log.id);
  },

  // Save learning log
  saveLearningLog() {
    const saveButton = document.getElementById("save-log");
    const logId = saveButton.getAttribute("data-log-id");

    const topic = document.getElementById("log-topic").value.trim();
    const content = document.getElementById("log-content").value.trim();
    const reflectionText = document
      .getElementById("log-reflection")
      .value.trim();

    if (!topic || !content) {
      this.showNotification("請填寫主題和內容", "warning");
      return;
    }

    // Process reflections
    let reflections = [];
    if (reflectionText) {
      reflections = reflectionText
        .split("\n")
        .filter((line) => line.trim() !== "");
    }

    const log = {
      id: logId || null, // null for new log, id for update
      topic,
      content,
      reflections,
    };

    this.showLoading("儲存日誌中...");

    api
      .saveLearningLog(log)
      .then((response) => {
        this.hideLoading();

        this.showNotification("日誌已成功儲存", "success");

        // Reset the form
        document.getElementById("log-topic").value = "";
        document.getElementById("log-content").value = "";
        document.getElementById("log-reflection").value = "";

        // Reset save button
        saveButton.textContent = "儲存日誌";
        saveButton.removeAttribute("data-log-id");

        // Reload logs
        this.loadLearningLogs();
      })
      .catch((error) => {
        this.hideLoading();
        console.error("Error saving log:", error);
        this.showNotification("儲存日誌時發生錯誤", "error");
      });
  },
};

// Add CSS for UI components
(function () {
  const style = document.createElement("style");
  style.textContent = `
          /* Notification styles */
          .notification-container {
              position: fixed;
              top: 20px;
              right: 20px;
              z-index: 1000;
              display: flex;
              flex-direction: column;
              gap: 10px;
          }
  
          .notification {
              padding: 12px 16px;
              border-radius: 4px;
              display: flex;
              align-items: center;
              justify-content: space-between;
              background-color: #fff;
              box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
              max-width: 350px;
              animation: notification-in 0.3s ease-out;
          }
  
          .notification.hiding {
              animation: notification-out 0.3s ease-in forwards;
          }
  
          .notification.info {
              border-left: 4px solid #3498db;
          }
  
          .notification.success {
              border-left: 4px solid #2ecc71;
          }
  
          .notification.warning {
              border-left: 4px solid #f39c12;
          }
  
          .notification.error {
              border-left: 4px solid #e74c3c;
          }
  
          .notification button {
              background: none;
              border: none;
              font-size: 18px;
              cursor: pointer;
              padding: 0 0 0 10px;
              color: #777;
          }
  
          @keyframes notification-in {
              0% {
                  transform: translateX(100%);
                  opacity: 0;
              }
              100% {
                  transform: translateX(0);
                  opacity: 1;
              }
          }
  
          @keyframes notification-out {
              0% {
                  transform: translateX(0);
                  opacity: 1;
              }
              100% {
                  transform: translateX(100%);
                  opacity: 0;
              }
          }
  
          /* Modal styles */
          .modal-container {
              position: fixed;
              top: 0;
              left: 0;
              width: 100%;
              height: 100%;
              background-color: rgba(0, 0, 0, 0.5);
              display: flex;
              align-items: center;
              justify-content: center;
              z-index: 1000;
          }
  
          .modal {
              background-color: #fff;
              border-radius: 8px;
              box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
              width: 90%;
              max-width: 600px;
              max-height: 90vh;
              overflow-y: auto;
          }
  
          .modal-content {
              display: flex;
              flex-direction: column;
          }
  
          .modal-header {
              display: flex;
              align-items: center;
              justify-content: space-between;
              padding: 15px 20px;
              border-bottom: 1px solid #e0e0e0;
          }
  
          .modal-body {
              padding: 20px;
              overflow-y: auto;
          }
  
          .modal-footer {
              padding: 15px 20px;
              border-top: 1px solid #e0e0e0;
              display: flex;
              justify-content: flex-end;
              gap: 10px;
          }
  
          .close-modal {
              font-size: 24px;
              cursor: pointer;
              color: #777;
          }
  
          /* Loading styles */
          .loading-container {
              position: fixed;
              top: 0;
              left: 0;
              width: 100%;
              height: 100%;
              background-color: rgba(255, 255, 255, 0.8);
              display: flex;
              align-items: center;
              justify-content: center;
              z-index: 2000;
              opacity: 0;
              pointer-events: none;
              transition: opacity 0.3s;
          }
  
          .loading-container.show {
              opacity: 1;
              pointer-events: auto;
          }
  
          .loading-indicator {
              background-color: #fff;
              padding: 20px;
              border-radius: 8px;
              box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
              text-align: center;
          }
  
          .spinner {
              width: 40px;
              height: 40px;
              margin: 0 auto 15px;
              border: 4px solid #f3f3f3;
              border-top: 4px solid #3498db;
              border-radius: 50%;
              animation: spin 1s linear infinite;
          }
  
          @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
          }
  
          /* Survey styles */
          .survey-question {
              margin-bottom: 20px;
              padding-bottom: 20px;
              border-bottom: 1px solid #e0e0e0;
          }
  
          .survey-question:last-child {
              border-bottom: none;
          }
  
          /* Analysis styles */
          .analysis-section {
              margin-bottom: 20px;
          }
  
          .analysis-section h4 {
              margin-bottom: 10px;
              color: #333;
          }
  
          .analysis-section ul {
              padding-left: 20px;
          }
  
          /* Button container */
          .button-container {
              display: flex;
              gap: 10px;
              margin-top: 20px;
          }
  
          /* Survey modal */
          .modal.show {
              display: block;
          }
      `;

  document.head.appendChild(style);
})();
