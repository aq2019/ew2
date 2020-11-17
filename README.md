This is a consulting project for [Empower Work](https://www.empowerwork.org). Due to confidentiality, data is not accessible from this repo.

## data

Data consists of archived conversations between texters and peer counselors. Each conversation includes multiple messages. Each message includes text as well as contextual information such as date and time,

Conversations are manually assigned with tags to signal *content* and *completion status*.

In particular:

- One or more **content** tags may be assigned to each conversation, such as:

  - 'Anxiety/Stress', 'Confidence', 'Stuck', 'Undervalued', 'Fear', 'Frustrated', 'Uncertain', 'Bullied'
  - 'Benefits or Leave', 'Communication', 'Discrimination', 'Harassment', 'Payroll or time issue', 'Performance', 'Workload/Hours', 'Job Search'
  - 'Manager', 'Coworker'

The bullet points above represent different categories of tags: *Emotion*, *Issue*, *People*.

- One **completion status** tag may be assigned to each conversation: 'Initiated only', 'Incomplete', 'Completed'.

## tasks

1. content tagging: 

- Build models to assign **content** tags to future conversations.

2. dashboard: 

- Visualize key metrics to track performance and answer key operational questions, such as conversations count, completion rate (over time and break down by content tags), topic frequency, message volume by day and hour, counselor activities and scheduling.

- Implement Machine Learning **completion status** classification for the completion rates in the dashboard.



