Submission for Conrnell AI hackathon 2025

Slides:

https://www.canva.com/design/DAGeiovlBww/2PX_CsDO6ej1_0NbFDt38A/edit?utm_content=DAGeiovlBww&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

Pitch:

Recording: https://www.canva.com/design/DAGeiovlBww/2PX_CsDO6ej1_0NbFDt38A/edit?utm_content=DAGeiovlBww&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

General Transcript: 

I want you to imagine you're a first generation college student, juggling jobs with schoolwork to pay off bills, but facing food insecurity as a result. Meanwhile, food banks pour money into ad campaigns and outreach, but fail to connect with the people who need them most. Speaking personally, I know about both sides of this problem, as both a former nonprofit outreach secretary and current RA trying to help my students. My team at EduConnect.AI wondered — how can we help?

So, who are we? We’re a group of 5 CS 2nd-years from Cornell — Andrew is our team leader and algorithm dev, Anthony, Jason, and Aaron are our frontend/fullstack developers, and Rishi is our pitch leader. 

	Going back to the problem of connecting nonprofits with students/schools who need their resources, we wanted to prioritize empathy through easy accessibility and well-fitting matches. We made EduConnect a web app with an in-built algorithm, utilizing StreamLit and React for the fullstack web app, SerpAPI for RAG facilitation, and Open AI’s LLM for prompt engineering and classification.
	Let me explain the pipeline. The user types in our chat what they’re looking for, like a food bank nonprofit around Cornell University in Ithaca. Open AI takes their prompt, filters on the back end for keywords (like food bank and Ithaca here), and searches real-time on SerpAPI. We internally filter for nonprofits by looking for the .org extension on links, and provide summaries of the organization’s operations/mission. Next is the fun part. So far, the user might as well google nonprofits for themselves. But with the power of our LLM-based matching algorithm, the users can “swipe” yes/no on nonprofits, helping our model learn more and create better internal searches. This allows users to ultimately find the nonprofit that can best help them, and conversely help nonprofits find users that really need their resources.

	EduConnect’s novel solution promises to beat nonprofit ad campaigns in price and match quality, a huge selling point. How can we make an ethical profit? One idea is a subscription plan, where nonprofits could pay to be “featured” on our page. Additionally, user-side banner ads could help generate revenue. Finally, we can offer personalized consulting services to nonprofits, utilizing our algorithm helping them improve their connections to target users and donors.
	
	Our MVP has a lot of promise, and we believe future iterations can more effectively fulfill our empathetic vision. From donor-side nonprofit matching, to beta-testing improved algorithms (including ones that source directly from nonprofit databases or that memoize past searches), to app components like messaging or dashboards, we look forward to working on EduConnect.AI more and watching it grow! Thanks for listening, and we’re open to any questions or comments now!
