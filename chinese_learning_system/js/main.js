// main.js - Main application initialization and event handlers

document.addEventListener("DOMContentLoaded", function () {
  // Initialize the application
  initializeApp();

  // Set up event listeners
  setupEventListeners();
});

// Initialize the application
function initializeApp() {
  // Load student profile if exists
  loadStudentProfile();

  // Update dashboard display
  updateDashboard();

  // Default to dashboard view
  showView("dashboard");
}

// Set up all event listeners
function setupEventListeners() {
  // Menu navigation
  document.querySelectorAll(".menu-item").forEach((item) => {
    item.addEventListener("click", function () {
      const view = this.getAttribute("data-view");
      showView(view);

      // Update active menu item
      document
        .querySelectorAll(".menu-item")
        .forEach((i) => i.classList.remove("active"));
      this.classList.add("active");
    });
  });

  // Profile related events
  document
    .getElementById("save-profile")
    .addEventListener("click", saveProfile);
  document
    .getElementById("start-survey")
    .addEventListener("click", showLearningStyleSurvey);
  document.querySelectorAll(".style-btn").forEach((btn) => {
    btn.addEventListener("click", function () {
      document
        .querySelectorAll(".style-btn")
        .forEach((b) => b.classList.remove("active"));
      this.classList.add("active");
    });
  });

  // Upload related events
  const dropArea = document.getElementById("drop-area");
  const fileInput = document.getElementById("file-input");

  document.getElementById("select-files").addEventListener("click", () => {
    fileInput.click();
  });

  fileInput.addEventListener("change", handleFileSelection);

  dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.classList.add("highlight");
  });

  dropArea.addEventListener("dragleave", () => {
    dropArea.classList.remove("highlight");
  });

  dropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    dropArea.classList.remove("highlight");

    const files = e.dataTransfer.files;
    handleFiles(files);
  });

  document
    .getElementById("process-uploads")
    .addEventListener("click", processUploads);

  // Test related events
  document
    .getElementById("start-pretest")
    .addEventListener("click", startPretest);
  document
    .getElementById("start-posttest")
    .addEventListener("click", startPosttest);

  // Discussion related events
  document
    .getElementById("discussion-topic")
    .addEventListener("change", loadDiscussionTopic);
  document
    .getElementById("send-message")
    .addEventListener("click", sendChatMessage);
  document
    .getElementById("message-input")
    .addEventListener("keypress", function (e) {
      if (e.key === "Enter") {
        sendChatMessage();
      }
    });

  // Learning logs related events
  document
    .getElementById("save-log")
    .addEventListener("click", saveLearninglog);

  // Modal close button
  document.querySelectorAll(".close-modal").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.getElementById("survey-modal").classList.remove("show");
    });
  });

  // Survey submission
  document
    .getElementById("submit-survey")
    .addEventListener("click", submitLearningStyleSurvey);
}

// Show the specified view and hide others
function showView(viewName) {
  document.querySelectorAll(".view").forEach((view) => {
    view.classList.add("hidden");
  });

  document.getElementById(`${viewName}-view`).classList.remove("hidden");

  // Additional view-specific initialization
  if (viewName === "learning") {
    loadLearningModules();
  } else if (viewName === "tests") {
    checkTestAvailability();
  } else if (viewName === "discussion") {
    loadDiscussionTopics();
  } else if (viewName === "logs") {
    loadLearningLogs();
  }
}

// Load student profile from server
function loadStudentProfile() {
  api
    .getStudentProfile()
    .then((profile) => {
      if (profile) {
        // Update UI with profile information
        document.getElementById("student-name").value = profile.name || "";

        if (profile.learning_style) {
          document.querySelectorAll(".style-btn").forEach((btn) => {
            if (btn.getAttribute("data-style") === profile.learning_style) {
              btn.classList.add("active");
            }
          });
        }

        document.getElementById("student-interests").value = profile.interests
          ? profile.interests.join(", ")
          : "";

        // Update dashboard
        document.getElementById("knowledge-level").textContent =
          profile.current_knowledge_level || "初學者";
        document.getElementById("learning-style").textContent =
          profile.learning_style || "未確定";

        // Enable/disable features based on profile
        updateFeatureAvailability(profile);
      }
    })
    .catch((error) => {
      console.error("Error loading profile:", error);
      ui.showNotification("載入學生檔案時發生錯誤", "error");
    });
}

// Save student profile
function saveProfile() {
  const name = document.getElementById("student-name").value;
  const learningStyleBtn = document.querySelector(".style-btn.active");
  const learningStyle = learningStyleBtn
    ? learningStyleBtn.getAttribute("data-style")
    : "";
  const interestsString = document.getElementById("student-interests").value;
  const interests = interestsString
    ? interestsString.split(",").map((i) => i.trim())
    : [];

  const profile = {
    name,
    learning_style: learningStyle,
    interests,
  };

  api
    .saveStudentProfile(profile)
    .then(() => {
      ui.showNotification("學生檔案已儲存成功", "success");
      updateDashboard();
    })
    .catch((error) => {
      console.error("Error saving profile:", error);
      ui.showNotification("儲存學生檔案時發生錯誤", "error");
    });
}

// Show learning style survey modal
function showLearningStyleSurvey() {
  api
    .generateLearningStyleSurvey()
    .then((survey) => {
      // Display survey in modal
      const surveyContent = document.getElementById("survey-content");
      surveyContent.innerHTML = "";

      // Assuming the server returns an array of questions
      survey.questions.forEach((question, index) => {
        const questionDiv = document.createElement("div");
        questionDiv.className = "survey-question";

        const questionText = document.createElement("p");
        questionText.className = "question-text";
        questionText.textContent = `${index + 1}. ${question.text}`;

        const optionsContainer = document.createElement("div");
        optionsContainer.className = "options-container";

        question.options.forEach((option) => {
          const optionItem = document.createElement("div");
          optionItem.className = "option-item";
          optionItem.setAttribute("data-value", option.value);
          optionItem.setAttribute("data-question", index);

          optionItem.textContent = option.text;

          optionItem.addEventListener("click", function () {
            const questionIndex = this.getAttribute("data-question");
            document
              .querySelectorAll(
                `.option-item[data-question="${questionIndex}"]`
              )
              .forEach((opt) => {
                opt.classList.remove("selected");
              });
            this.classList.add("selected");
          });

          optionsContainer.appendChild(optionItem);
        });

        questionDiv.appendChild(questionText);
        questionDiv.appendChild(optionsContainer);
        surveyContent.appendChild(questionDiv);
      });

      // Show the modal
      document.getElementById("survey-modal").classList.add("show");
    })
    .catch((error) => {
      console.error("Error generating survey:", error);
      ui.showNotification("生成學習風格問卷時發生錯誤", "error");
    });
}

// Submit learning style survey
function submitLearningStyleSurvey() {
  const selectedOptions = [];
  document.querySelectorAll(".option-item.selected").forEach((option) => {
    selectedOptions.push({
      questionIndex: parseInt(option.getAttribute("data-question")),
      value: option.getAttribute("data-value"),
    });
  });

  if (
    selectedOptions.length <
    document.querySelectorAll(".survey-question").length
  ) {
    ui.showNotification("請回答所有問題", "warning");
    return;
  }

  api
    .submitLearningStyleSurvey(selectedOptions)
    .then((result) => {
      // Hide the modal
      document.getElementById("survey-modal").classList.remove("show");

      // Update UI with learning style result
      const learningStyle = result.learning_style;
      document.getElementById("learning-style").textContent = learningStyle;

      document.querySelectorAll(".style-btn").forEach((btn) => {
        btn.classList.remove("active");
        if (btn.getAttribute("data-style") === learningStyle) {
          btn.classList.add("active");
        }
      });

      ui.showNotification(`學習風格確定為：${learningStyle}`, "success");

      // Update dashboard and features
      updateDashboard();
      loadStudentProfile();
    })
    .catch((error) => {
      console.error("Error submitting survey:", error);
      ui.showNotification("提交問卷時發生錯誤", "error");
    });
}

// Handle file selection from the input
function handleFileSelection(e) {
  const files = e.target.files;
  handleFiles(files);
}

// Process the selected files
function handleFiles(files) {
  if (files.length === 0) return;

  const uploadList = document.getElementById("upload-list");

  // Clear the list if it's the first upload
  if (uploadList.childElementCount === 0) {
    uploadList.innerHTML = "";
  }

  // Add files to the list
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    if (file.type !== "application/pdf") {
      ui.showNotification("只能上傳PDF檔案", "warning");
      continue;
    }

    const listItem = document.createElement("li");
    listItem.className = "upload-list-item";
    listItem.setAttribute("data-file-name", file.name);

    const fileInfo = document.createElement("div");
    fileInfo.className = "file-info";

    const icon = document.createElement("i");
    icon.className = "fas fa-file-pdf";

    const fileName = document.createElement("span");
    fileName.textContent = file.name;

    fileInfo.appendChild(icon);
    fileInfo.appendChild(fileName);

    const status = document.createElement("span");
    status.className = "file-status";
    status.textContent = "準備上傳";

    listItem.appendChild(fileInfo);
    listItem.appendChild(status);

    uploadList.appendChild(listItem);
  }

  // Enable the process button
  document.getElementById("process-uploads").disabled = false;
}

// Process the uploaded files
function processUploads() {
  const uploadList = document.getElementById("upload-list");
  const fileItems = uploadList.querySelectorAll(".upload-list-item");

  if (fileItems.length === 0) {
    ui.showNotification("請先選擇檔案", "warning");
    return;
  }

  ui.showNotification("開始處理檔案...", "info");

  // Get the file input element
  const fileInput = document.getElementById("file-input");
  const files = fileInput.files;

  // Create a FormData object to send files
  const formData = new FormData();
  for (let i = 0; i < files.length; i++) {
    formData.append("files", files[i]);
  }

  // Update status for all files
  fileItems.forEach((item) => {
    const statusElement = item.querySelector(".file-status");
    statusElement.textContent = "上傳中...";
  });

  // Disable the process button
  document.getElementById("process-uploads").disabled = true;

  // Upload the files to the server
  api
    .uploadDocuments(formData)
    .then((response) => {
      if (response.success) {
        // Update status for successful uploads
        fileItems.forEach((item) => {
          const fileName = item.getAttribute("data-file-name");
          const statusElement = item.querySelector(".file-status");

          if (response.processed.includes(fileName)) {
            statusElement.textContent = "處理完成";
            statusElement.classList.add("success");
          } else if (response.failed.includes(fileName)) {
            statusElement.textContent = "處理失敗";
            statusElement.classList.add("error");
          }
        });

        ui.showNotification("檔案處理完成", "success");

        // Reset file input
        fileInput.value = "";

        // Update available modules
        if (response.processed.length > 0) {
          updateFeatureAvailability();
        }
      } else {
        throw new Error("處理檔案時發生錯誤");
      }
    })
    .catch((error) => {
      console.error("Error processing uploads:", error);
      ui.showNotification("處理上傳檔案時發生錯誤", "error");

      // Update status for all files
      fileItems.forEach((item) => {
        const statusElement = item.querySelector(".file-status");
        statusElement.textContent = "處理失敗";
        statusElement.classList.add("error");
      });
    });
}

// Update dashboard with latest information
function updateDashboard() {
  api
    .getDashboardData()
    .then((data) => {
      // Update knowledge level and learning style
      document.getElementById("knowledge-level").textContent =
        data.knowledge_level || "初學者";
      document.getElementById("learning-style").textContent =
        data.learning_style || "未確定";

      // Update progress
      const progress = data.progress || 0;
      document.querySelector(".progress").style.width = `${progress}%`;
      document.querySelector(
        ".progress-text"
      ).textContent = `${progress}% 完成`;

      // Update recent activities
      const activitiesList = document.getElementById("recent-activities");
      activitiesList.innerHTML = "";

      if (data.recent_activities && data.recent_activities.length > 0) {
        data.recent_activities.forEach((activity) => {
          const li = document.createElement("li");
          li.textContent = `${activity.date}: ${activity.description}`;
          activitiesList.appendChild(li);
        });
      } else {
        const li = document.createElement("li");
        li.textContent = "尚無學習活動";
        activitiesList.appendChild(li);
      }

      // Update recommendations
      const recommendationsContainer = document.getElementById(
        "learning-recommendations"
      );
      recommendationsContainer.innerHTML = "";

      if (data.recommendations && data.recommendations.length > 0) {
        data.recommendations.forEach((recommendation) => {
          const p = document.createElement("p");
          p.textContent = recommendation;
          recommendationsContainer.appendChild(p);
        });
      } else {
        const p = document.createElement("p");
        p.textContent = "完成學習風格問卷以獲取個人化推薦";
        recommendationsContainer.appendChild(p);
      }
    })
    .catch((error) => {
      console.error("Error updating dashboard:", error);
    });
}

// Check test availability
function checkTestAvailability() {
  api
    .getTestAvailability()
    .then((data) => {
      document.getElementById("start-pretest").disabled =
        !data.pretest_available;
      document.getElementById("start-posttest").disabled =
        !data.posttest_available;
    })
    .catch((error) => {
      console.error("Error checking test availability:", error);
    });
}

// Start pretest
function startPretest() {
  api
    .generatePretest()
    .then((test) => {
      renderTest(test, "pretest");
    })
    .catch((error) => {
      console.error("Error generating pretest:", error);
      ui.showNotification("生成前測時發生錯誤", "error");
    });
}

// Start posttest
function startPosttest() {
  api
    .generatePosttest()
    .then((test) => {
      renderTest(test, "posttest");
    })
    .catch((error) => {
      console.error("Error generating posttest:", error);
      ui.showNotification("生成後測時發生錯誤", "error");
    });
}

// Render test in the UI
function renderTest(test, testType) {
  const testContent = document.getElementById("test-content");
  testContent.innerHTML = "";

  // Create test header
  const header = document.createElement("div");
  header.className = "test-header";

  const title = document.createElement("h3");
  title.textContent = test.title;

  const description = document.createElement("p");
  description.textContent = test.description;

  header.appendChild(title);
  header.appendChild(description);
  testContent.appendChild(header);

  // Create questions
  test.questions.forEach((question, index) => {
    const questionContainer = document.createElement("div");
    questionContainer.className = "question-container";
    questionContainer.setAttribute("data-question-index", index);

    const questionText = document.createElement("div");
    questionText.className = "question-text";
    questionText.textContent = `${index + 1}. ${question.question}`;

    const optionsContainer = document.createElement("div");
    optionsContainer.className = "options-container";

    question.choices.forEach((choice, choiceIndex) => {
      const option = document.createElement("div");
      option.className = "option-item";
      option.setAttribute("data-choice-index", choiceIndex);
      option.textContent = choice;

      option.addEventListener("click", function () {
        // If already submitted, do nothing
        if (
          this.classList.contains("correct") ||
          this.classList.contains("incorrect")
        ) {
          return;
        }

        const questionIndex = parseInt(
          this.parentElement.parentElement.getAttribute("data-question-index")
        );

        document
          .querySelectorAll(
            `.question-container[data-question-index="${questionIndex}"] .option-item`
          )
          .forEach((opt) => {
            opt.classList.remove("selected");
          });

        this.classList.add("selected");
      });

      optionsContainer.appendChild(option);
    });

    questionContainer.appendChild(questionText);
    questionContainer.appendChild(optionsContainer);
    testContent.appendChild(questionContainer);
  });

  // Add submit button
  const submitButton = document.createElement("button");
  submitButton.className = "primary-btn";
  submitButton.textContent = "提交測驗";
  submitButton.addEventListener("click", () => submitTest(test, testType));

  testContent.appendChild(submitButton);
}

// Submit test answers
function submitTest(test, testType) {
  const questionContainers = document.querySelectorAll(".question-container");
  const answers = [];
  let allAnswered = true;

  questionContainers.forEach((container) => {
    const questionIndex = parseInt(
      container.getAttribute("data-question-index")
    );
    const selectedOption = container.querySelector(".option-item.selected");

    if (selectedOption) {
      const choiceIndex = parseInt(
        selectedOption.getAttribute("data-choice-index")
      );
      answers.push({
        questionIndex,
        choiceIndex,
        choiceText: test.questions[questionIndex].choices[choiceIndex],
      });
    } else {
      allAnswered = false;
    }
  });

  if (!allAnswered) {
    ui.showNotification("請回答所有問題", "warning");
    return;
  }

  // Submit answers to server
  api
    .submitTestAnswers(testType, answers)
    .then((result) => {
      // Show results
      showTestResults(result, test);

      // Update dashboard with new progress
      updateDashboard();

      // Update test availability
      checkTestAvailability();
    })
    .catch((error) => {
      console.error("Error submitting test:", error);
      ui.showNotification("提交測驗時發生錯誤", "error");
    });
}

// Show test results
function showTestResults(result, test) {
  const testContent = document.getElementById("test-content");

  // Mark correct and incorrect answers
  result.answers.forEach((answer) => {
    const questionContainer = document.querySelector(
      `.question-container[data-question-index="${answer.questionIndex}"]`
    );
    const options = questionContainer.querySelectorAll(".option-item");

    // Mark the correct answer
    options[answer.correctChoiceIndex].classList.add("correct");

    // If user's answer is incorrect, mark it
    if (answer.choiceIndex !== answer.correctChoiceIndex) {
      options[answer.choiceIndex].classList.add("incorrect");
    }

    // Add explanation
    const explanation = document.createElement("div");
    explanation.className = "explanation";
    explanation.textContent =
      answer.explanation || test.questions[answer.questionIndex].explanation;

    questionContainer.appendChild(explanation);
  });

  // Create results summary
  const resultsContainer = document.createElement("div");
  resultsContainer.className = "test-results";

  const scoreElement = document.createElement("div");
  scoreElement.className = "test-score";
  scoreElement.textContent = `得分：${result.score}/${
    result.total
  } (${Math.round(result.percentage)}%)`;

  const levelElement = document.createElement("p");
  levelElement.textContent = `知識水平評估：${result.knowledge_level}`;

  const feedbackElement = document.createElement("p");
  feedbackElement.textContent = result.feedback;

  resultsContainer.appendChild(scoreElement);
  resultsContainer.appendChild(levelElement);
  resultsContainer.appendChild(feedbackElement);

  // Add next steps if provided
  if (result.next_steps && result.next_steps.length > 0) {
    const nextStepsTitle = document.createElement("h4");
    nextStepsTitle.textContent = "建議下一步";
    resultsContainer.appendChild(nextStepsTitle);

    const nextStepsList = document.createElement("ul");
    result.next_steps.forEach((step) => {
      const li = document.createElement("li");
      li.textContent = step;
      nextStepsList.appendChild(li);
    });

    resultsContainer.appendChild(nextStepsList);
  }

  // Replace the submit button with results
  const submitButton = testContent.querySelector(".primary-btn");
  if (submitButton) {
    submitButton.remove();
  }

  testContent.appendChild(resultsContainer);

  // Scroll to results
  resultsContainer.scrollIntoView({ behavior: "smooth" });
}

// Load learning modules
function loadLearningModules() {
  api
    .getLearningModules()
    .then((modules) => {
      const modulesContainer = document.getElementById("modules-container");
      modulesContainer.innerHTML = "";

      if (modules && modules.length > 0) {
        modules.forEach((module, index) => {
          const moduleItem = document.createElement("div");
          moduleItem.className = "module-item";
          moduleItem.setAttribute("data-module-id", module.id);

          const title = document.createElement("h4");
          title.textContent = module.title;

          const description = document.createElement("p");
          description.textContent = module.description;

          moduleItem.appendChild(title);
          moduleItem.appendChild(description);

          moduleItem.addEventListener("click", function () {
            // Update selected module
            document.querySelectorAll(".module-item").forEach((item) => {
              item.classList.remove("active");
            });
            this.classList.add("active");

            // Load module content
            loadModuleContent(module.id);
          });

          modulesContainer.appendChild(moduleItem);

          // Select the first module by default
          if (index === 0) {
            moduleItem.click();
          }
        });
      } else {
        const emptyState = document.createElement("p");
        emptyState.className = "empty-state";
        emptyState.textContent = "尚無可用模組。請先上傳教材或完成前測。";
        modulesContainer.appendChild(emptyState);
      }
    })
    .catch((error) => {
      console.error("Error loading modules:", error);
      ui.showNotification("載入學習模組時發生錯誤", "error");
    });
}

// Load module content
function loadModuleContent(moduleId) {
  api
    .getModuleContent(moduleId)
    .then((content) => {
      const contentDisplay = document.getElementById("content-display");

      // Use marked.js to render markdown
      contentDisplay.innerHTML = marked.parse(content.content);

      // Apply syntax highlighting if there are code blocks
      document.querySelectorAll("pre code").forEach((block) => {
        hljs.highlightElement(block);
      });
    })
    .catch((error) => {
      console.error("Error loading module content:", error);
      ui.showNotification("載入模組內容時發生錯誤", "error");
    });
}

// Load discussion topics
function loadDiscussionTopics() {
  api
    .getDiscussionTopics()
    .then((topics) => {
      const topicSelect = document.getElementById("discussion-topic");

      // Clear existing options
      while (topicSelect.options.length > 1) {
        topicSelect.remove(1);
      }

      if (topics && topics.length > 0) {
        topics.forEach((topic) => {
          const option = document.createElement("option");
          option.value = topic.id;
          option.textContent = topic.title;
          topicSelect.appendChild(option);
        });

        topicSelect.disabled = false;
      } else {
        const option = document.createElement("option");
        option.textContent = "尚無可用討論主題";
        topicSelect.appendChild(option);

        topicSelect.disabled = true;
      }
    })
    .catch((error) => {
      console.error("Error loading discussion topics:", error);
      ui.showNotification("載入討論主題時發生錯誤", "error");
    });
}

// Load discussion for a specific topic
function loadDiscussionTopic() {
  const topicId = document.getElementById("discussion-topic").value;

  if (!topicId) return;

  // Clear existing messages
  const messagesContainer = document.getElementById("chat-messages");
  messagesContainer.innerHTML = "";

  // Add welcome message
  const welcomeMessage = document.createElement("div");
  welcomeMessage.className = "chat-message system";
  welcomeMessage.innerHTML = "<p>載入對話中，請稍候...</p>";
  messagesContainer.appendChild(welcomeMessage);

  // Enable chat input
  document.getElementById("message-input").disabled = false;
  document.getElementById("send-message").disabled = false;

  // Get topic information
  api
    .getDiscussionTopic(topicId)
    .then((topic) => {
      // Update welcome message
      welcomeMessage.innerHTML = `<p>歡迎來到「${topic.title}」的討論。您可以向AI學習夥伴提問或討論相關主題。</p>`;

      // Add initial AI message
      const aiMessage = document.createElement("div");
      aiMessage.className = "chat-message assistant";
      aiMessage.innerHTML = `<p>${
        topic.initial_message ||
        "嗨！我是您的學習夥伴。您有什麼問題或想討論什麼？"
      }</p>`;
      messagesContainer.appendChild(aiMessage);

      // Scroll to bottom
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    })
    .catch((error) => {
      console.error("Error loading discussion topic:", error);
      ui.showNotification("載入討論主題時發生錯誤", "error");

      welcomeMessage.innerHTML = "<p>載入討論主題時發生錯誤，請稍後再試。</p>";
    });
}

// This is the continuation of the main.js file, completing what was cut off

// Send chat message (completion of the function)
function sendChatMessage() {
  const messageInput = document.getElementById("message-input");
  const message = messageInput.value.trim();

  if (!message) return;

  const topicId = document.getElementById("discussion-topic").value;
  const messagesContainer = document.getElementById("chat-messages");

  // Add user message to chat
  const userMessage = document.createElement("div");
  userMessage.className = "chat-message user";
  userMessage.innerHTML = `<p>${message}</p>`;
  messagesContainer.appendChild(userMessage);

  // Clear input
  messageInput.value = "";

  // Disable input while waiting for response
  messageInput.disabled = true;
  document.getElementById("send-message").disabled = true;

  // Add loading message
  const loadingMessage = document.createElement("div");
  loadingMessage.className = "chat-message assistant loading";
  loadingMessage.innerHTML = "<p>思考中...</p>";
  messagesContainer.appendChild(loadingMessage);

  // Scroll to bottom
  messagesContainer.scrollTop = messagesContainer.scrollHeight;

  // Send message to server
  api
    .sendChatMessage(topicId, message)
    .then((response) => {
      // Remove loading message
      loadingMessage.remove();

      // Add AI response
      const aiMessage = document.createElement("div");
      aiMessage.className = "chat-message assistant";
      aiMessage.innerHTML = `<p>${response.message}</p>`;
      messagesContainer.appendChild(aiMessage);

      // Re-enable input
      messageInput.disabled = false;
      document.getElementById("send-message").disabled = false;
      messageInput.focus();

      // Scroll to bottom
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    })
    .catch((error) => {
      console.error("Error sending message:", error);

      // Remove loading message
      loadingMessage.remove();

      // Add error message
      const errorMessage = document.createElement("div");
      errorMessage.className = "chat-message system";
      errorMessage.innerHTML = "<p>發送訊息時發生錯誤，請稍後再試。</p>";
      messagesContainer.appendChild(errorMessage);

      // Re-enable input
      messageInput.disabled = false;
      document.getElementById("send-message").disabled = false;
      messageInput.focus();
    });
}

// Load learning logs
function loadLearningLogs() {
  ui.loadLearningLogs();
}

// Save learning log
function saveLearninglog() {
  ui.saveLearningLog();
}

// Update feature availability based on user progress
function updateFeatureAvailability(profile) {
  ui.updateFeatureAvailability(profile);
}

// Add modal window CSS for survey
document.addEventListener("DOMContentLoaded", function () {
  const style = document.createElement("style");
  style.textContent = `
          .modal {
              display: none;
              position: fixed;
              z-index: 1000;
              left: 0;
              top: 0;
              width: 100%;
              height: 100%;
              overflow: auto;
              background-color: rgba(0, 0, 0, 0.5);
          }
          
          .modal.show {
              display: flex;
              align-items: center;
              justify-content: center;
          }
          
          .modal-content {
              background-color: #fff;
              margin: auto;
              padding: 20px;
              border-radius: 8px;
              box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
              width: 80%;
              max-width: 600px;
              max-height: 80vh;
              overflow-y: auto;
          }
          
          .modal-header {
              display: flex;
              align-items: center;
              justify-content: space-between;
              padding-bottom: 15px;
              border-bottom: 1px solid #e0e0e0;
              margin-bottom: 15px;
          }
          
          .close-modal {
              font-size: 24px;
              font-weight: bold;
              cursor: pointer;
          }
          
          .modal-footer {
              display: flex;
              justify-content: flex-end;
              padding-top: 15px;
              border-top: 1px solid #e0e0e0;
              margin-top: 15px;
          }
      `;
  document.head.appendChild(style);
});
