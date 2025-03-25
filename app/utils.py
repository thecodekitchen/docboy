import logging
import asyncio
import json

async def stream_callable(cb: callable, args: list, kwargs: dict, logger: logging.Logger):
    try:
        counter = 0
        logger.info("Starting stream generation")
        
        async for chunk in cb(*args, **kwargs):
            counter += 1
            logger.debug(f"Chunk {counter}: {chunk}")
            
            # Format for SSE
            yield f"data: {chunk}\n\n"
            
            # Add small delay for visibility in logs
            await asyncio.sleep(0.1)
            
        logger.info(f"Stream complete, sent {counter} chunks")
        
    except Exception as e:
        logger.error(f"Error in stream generation: {str(e)}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"