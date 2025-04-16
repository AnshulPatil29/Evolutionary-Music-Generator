pipeline {
    agent any
    triggers {
        pollSCM('H/5 * * * *') // Poll the repository every 5 minutes
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
        stage('Clean Up Containers on Port 8501') {
            steps {
                script {
                    // This Windows batch block will loop through any container using port 8501
                    bat '''
                        @echo off
                        for /F "tokens=*" %%i in ('docker ps --filter "publish=8501" -q') do (
                            echo Removing container %%i which is using port 8501...
                            docker rm -f %%i
                        )
                    '''
                }
            }
        }
        stage('Run Docker Container') {
            steps {
                script {
                    // Run the new container, mapping port 8501 from container to host
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