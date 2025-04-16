pipeline {
    agent any
    triggers {
        pollSCM('H/5 * * * *') // Poll every 5 minutes
    }
    environment {
        IMAGE_NAME = "evolutionary-music-generator"
        CONTAINER_NAME = "music-generator-container"
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
                    docker.build("${IMAGE_NAME}:${env.BUILD_NUMBER}")
                }
            }
        }
        stage('Run Docker Container') {
            steps {
                script {
                    // Use bat for Windows commands instead of sh
                    bat "docker rm -f ${CONTAINER_NAME} || echo No existing container to remove"
                    bat "docker run -d --name ${CONTAINER_NAME} -p 8501:8501 ${IMAGE_NAME}:${env.BUILD_NUMBER}"
                }
            }
        }
    }
    post {
        always {
            echo 'Pipeline execution completed.'
        }
    }
}