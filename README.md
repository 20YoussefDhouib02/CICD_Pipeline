# CI/CD Pipeline for P2M Monitoring

This repository provides a robust CI/CD pipeline configuration to automate the building, testing, and deployment of applications. The pipeline is designed to ensure faster, more reliable software delivery while maintaining high-quality standards and enabling real-time monitoring.

## Project Overview

The P2M Monitoring project is structured into the following main components:

1. **Backend**: Contains the core logic and APIs for the application.
   - Built using Java with Spring Boot.
   - Includes a `Dockerfile` for containerization.
   - Configuration files are located in `src/main/resources`.

2. **Frontend**: Implements the user interface.
   - Built with Angular.
   - Key files include `angular.json`, `package.json`, and `src/index.html`.

3. **Monitoring**: Provides monitoring and alerting capabilities.
   - Uses Prometheus and Alertmanager for monitoring.
   - Configuration files include `prometheus.yml` and `alertmanager.yml`.

4. **VM_Setup**: Contains scripts and resources for setting up virtual machines and additional tools.
   - Includes anomaly detection scripts, rollback tools, and project-related resources.

## CI/CD Pipeline Features

- **Continuous Integration**:
  - Automated testing of code changes using unit and integration tests.
  - Linting and static code analysis to ensure code quality.
  - Build automation using Docker and Maven for the backend, and npm for the frontend.

- **Continuous Deployment**:
  - Deployment to staging and production environments using Docker Compose.
  - Versioned releases for rollback support.
  - Environment-specific configurations for seamless deployment.

- **Monitoring and Alerts**:
  - Real-time monitoring of application health using Prometheus.
  - Alerting for critical issues using Alertmanager.
  - Visualization of metrics and logs for debugging and performance analysis.

## VM_Setup Folder Details

The `VM_Setup` folder contains essential scripts and resources for setting up and managing virtual machines, as well as additional tools for anomaly detection and rollback mechanisms. Below is a detailed breakdown:

- **Anomaly Detection**:
  - `anomaly Detector script.py`: Python script for detecting anomalies in data.
  - `OLD anomaly detector.ipynb`: Jupyter Notebook for anomaly detection experiments.
  - `anomaly_plots/`: Contains visualizations of anomaly detection results, such as:
    - `anomaly_detector_job_iforest_scores.png`
    - `anomaly_detector_job_lstm_errors.png`
    - `anomaly_detector_job_prophet_forecast.png`

- **Rollback Tools**:
  - `Github-rollback service.txt`: Configuration for automating GitHub rollbacks.
  - `github-rollback-tool/`: Contains the rollback tool implementation:
    - `app.py`: Main application script.
    - `templates/`: HTML templates for the rollback tool interface.

- **Project Resources**:
  - `proj.puml`: UML diagram for project architecture.
  - `proj/`: Contains project-related images, such as `proj.png`.

## Prerequisites

- Docker and Docker Compose installed.
- Java Development Kit (JDK) for backend development.
- Node.js and npm for frontend development.
- Python and Jupyter Notebook for anomaly detection scripts.

## Getting Started

### Clone the Repository
```bash
git clone <repository-url>
cd p2m_cicdmonitoring
```

### Build and Run the Application

#### Using Docker Compose
```bash
docker-compose up --build
```

#### Manually

1. **Backend**:
   ```bash
   cd backend
   ./mvnw spring-boot:run
   ```

2. **Frontend**:
   ```bash
   cd frontend
   npm install
   npm start
   ```

3. **Monitoring**:
   - Start Prometheus and Alertmanager using their respective configuration files.

4. **VM_Setup**:
   - Run the anomaly detection script:
     ```bash
     python VM_Setup/anomaly Detector script.py
     ```
   - Use the rollback tool by navigating to `github-rollback-tool/` and running:
     ```bash
     python app.py
     ```

## CI/CD Pipeline Workflow

1. **Code Commit**:
   - Developers push code changes to the Git repository.
   - Triggers the CI pipeline.

2. **Build and Test**:
   - Backend: Maven builds the application and runs unit tests.
   - Frontend: npm builds the application and runs tests.

3. **Dockerization**:
   - Docker images are built for the backend and frontend.
   - Images are tagged with the commit hash or version number.

4. **Deployment**:
   - Docker Compose deploys the application to the staging environment.
   - After approval, the application is deployed to production.

5. **Monitoring**:
   - Prometheus collects metrics from the application.
   - Alertmanager sends notifications for critical issues.

## Folder Structure

- `backend/`: Contains the backend application code.
- `frontend/`: Contains the frontend application code.
- `monitoring/`: Contains monitoring configuration files.
- `VM_Setup/`: Scripts and resources for virtual machine setup.
- `docker-compose.yml`: Defines the services and configurations for the CI/CD pipeline.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push them to your fork.
4. Submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Resources

- [Docker Documentation](https://docs.docker.com/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Angular Documentation](https://angular.io/docs)
- [Spring Boot Documentation](https://spring.io/projects/spring-boot)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
