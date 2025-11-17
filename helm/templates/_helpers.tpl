{{/*
Return the full name for naming resources
*/}}
{{- define "facial-emotion-recognition.fullname" -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end }}

{{/*
Return the chart name
*/}}
{{- define "facial-emotion-recognition.name" -}}
{{- printf "%s" .Chart.Name -}}
{{- end }}
