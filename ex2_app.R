# GAM model for the prediction of onset of diabetes
library(mgcv)
library(ggplot2)
library(shiny)

data=read.csv('diabetes.csv')
# use all the data to train gam model, First set up the formula
vars=setdiff(colnames(data),"Outcome")
form=as.formula("Outcome==1~Pregnancies+s(Glucose)+BloodPressure+
                SkinThickness+Insulin+s(BMI)+s(DiabetesPedigreeFunction)+
                s(Age)")
model=gam(form,data = data, family = binomial(link="logit"))

ui <- fluidPage(
  titlePanel("Prediction of onset of diabetes"),
  sidebarLayout(
    sidebarPanel(p("Pregnancies: Number of times pregnant"),
                 br(),
                 p("Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test"),
                 br(),
                 p("BloodPressure: Diastolic blood pressure (mm Hg)"),
                 br(),
                 p("SkinThickness: Triceps skin fold thickness (mm)"),
                 br(),
                 p("Insulin: 2-Hour serum insulin (mu U/ml)"),
                 br(),
                 p("BMI: Body Mass Index"),
                 br(),
                 p("DiabetesPedigreeFunction: Diabetes pedigree function"),
                 br(),
                 p("Age: years")),
    mainPanel(p("Try manipulating the values of predictors below and observe how the changes
              in variables affect the outcome of prediction (probability). Remember that in
              the previous discussion, we had indentified", strong("Glucose, BMI and Age"),
              "as important variables and they were placed in the left-hand panel. Brackets
              beside the labels of each feature are the ranges of value of data used to
              train the model. Inserting values beyond the range could yield misleading
              result."),
              br(),
              br()
              )
  ),

  fluidRow(
    column(4, 
           "features set 1",
           numericInput("Glucose", label = "Glucose (0-199)", value = median(data$Glucose), min = 1,max=199),
           numericInput("BMI", label = "BMI (0-67.1)", value = median(data$BMI),min=0,max=67.1),
           numericInput("Age", label = "Age (21-81)", value = median(data$Age), min = 21, max=81, step=1)
    ),
    column(4, 
           "features set 2",
           numericInput("Pregnancies", label = "Pregnancies (0-17)", value = median(data$Pregnancies), min = 0, max=17, step = 1),
           numericInput("BloodPressure", label = "Blood Pressure (0-122)", value = median(data$BloodPressure), min=0, max=122, step = 1),
           numericInput("DiabetesPedigree", label = "Diabetes Pedigree Function (0.078-2.42)", value = median(data$DiabetesPedigreeFunction), 
                        min = 0.078, max= 2.42)
    ),
    column(4,
           "features set 3",
           numericInput("Insulin", label = "Insulin (0-846)", value = median(data$Insulin), min=0, max=846),
           numericInput("SkinThickness", label = "Skin Thickness (0-99)", value = median(data$SkinThickness), min=0, max=99)
    )
  ),
  fluidRow(
    column(12, plotOutput("bar"))
  )
)

server <- function(input, output, session) {
  df=reactive({
    newdata=data.frame(Glucose=input$Glucose,BMI=input$BMI,Age=input$Age,
                                           Pregnancies=input$Pregnancies,BloodPressure=input$BloodPressure,
                                           DiabetesPedigreeFunction=input$DiabetesPedigree,
                                           Insulin=input$Insulin,SkinThickness=input$SkinThickness)
    pred=round(predict(model,newdata = newdata,type="response"),4)
    data.frame(class=c("non-diabetic","diabetic"),probability=c(1-pred,pred))
  })
  output$bar=renderPlot({
    ggplot(data=df(),aes(x=class,y=probability,fill=class)) +
      geom_bar(stat="identity") +
      geom_text(aes(label=probability),vjust=1.6,color="white",size=5) +
      theme_minimal() +
      theme(axis.text.x = element_text(size = 25),axis.title = element_text(size=20))+
      ylim(0,1)
  })
}

shinyApp(ui, server)