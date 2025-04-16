pipeline {
    agent any
    environment {
        IMAGE_NAME = "evolutionary-music-generator"
        CONTAINER_NAME = "music-generator-container"
        TEST_REPORT_FILE = 'test-results.xml'
    }
    stages {
        stage('Checkout') {
            steps {
                git url: 'https://github.com/AnshulPatil29/Evolutionary-Music-Generator.git', branch: 'main'
            }
        }
        stage('Build Docker Image') {
            steps {
                script {
                    echo "Building Docker image: ${IMAGE_NAME}:${env.BUILD_NUMBER}"
                    docker.build("${IMAGE_NAME}:${env.BUILD_NUMBER}")
                }
            }
        }
        stage('Test') {
            steps {
                script {
                    echo "Running tests using image ${IMAGE_NAME}:${env.BUILD_NUMBER}"
                    bat """
                        docker run --rm ^
                          -v "%WORKSPACE%:/app" ^
                          ${IMAGE_NAME}:${env.BUILD_NUMBER} ^
                          pytest /app/tests --junitxml=/app/${TEST_REPORT_FILE}
                    """
                }
            }
            post {
                always {
                    echo "Archiving test results from ${TEST_REPORT_FILE}..."
                    junit "${TEST_REPORT_FILE}"
                }
            }
        }
        stage('Clean Up Existing Container') {
            steps {
                script {
                    echo "Attempting to remove any existing container named '${CONTAINER_NAME}'..."
                    bat '''
                        @echo off
                        docker rm -f %CONTAINER_NAME%
                        echo Container removal attempt finished.
                        exit /B 0
                    '''
                }
            }
        }
        stage('Run Docker Container') {
            steps {
                script {
                    echo "Running new container '${CONTAINER_NAME}' from image ${IMAGE_NAME}:${env.BUILD_NUMBER}"
                    bat "docker run -d --name %CONTAINER_NAME% -p 8501:8501 ${IMAGE_NAME}:${env.BUILD_NUMBER}"
                }
            }
        }
    }
    post {
        always {
            echo 'Pipeline execution finished.'
        }
        success {
            echo "Pipeline finished successfully for build ${env.BUILD_NUMBER}."
        }
        failure {
            echo "Pipeline failed on build ${env.BUILD_NUMBER}."
        }
    }
}