def gpt_generation(client, prompt, model_name='gpt-3.5-turbo-16k'):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}],
        n=1,
        stream=False,
        temperature=0.0,
        max_tokens=2000,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return response.choices[0].message.content
