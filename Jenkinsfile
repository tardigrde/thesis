pipeline {
  agent any
  stages {
    stage('pull changes') {
      steps {
        git(url: 'https://github.com/tardigrde/thesis', branch: 'constacc', changelog: true, poll: true)
      }
    }
  }
}