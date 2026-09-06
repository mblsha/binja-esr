import type { Handle } from '@sveltejs/kit';

// The script worker uses a shared RPC mailbox for existing synchronous
// register/peek APIs. These headers are also required from production proxies.
export const handle: Handle = async ({ event, resolve }) => {
	const response = await resolve(event);
	response.headers.set('Cross-Origin-Opener-Policy', 'same-origin');
	response.headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
	return response;
};
