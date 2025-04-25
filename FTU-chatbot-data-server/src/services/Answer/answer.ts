import { data } from '@/utils/data';
/* eslint-disable no-param-reassign */
/* eslint-disable no-underscore-dangle */
/* eslint-disable @typescript-eslint/naming-convention */
import axios from '@/utils/axios';
import { ip } from '@/utils/ip';


export interface IPayload {
  intent_id: string,
  page: number,
  limit: number,
  cond: object,
}

class Answer<T> {
  name: string;
  url: string;

  constructor({ name, url }: { name: string; url: string }) {
    this.name = name;
    this.url = url || name;
  }

  del = async (id: string) => {
    return axios.delete(`${ip}/${this.url}`, { params: { id } });
  };
  delWithImage = async (id: string) => {
    return axios.delete(`${ip}/files/images`, { params: { id } });
  };

  get = async (payload: IPayload) => {
    // console.log(payload);
    return axios.get(`${ip}/sort_answer`, { params: payload });
  };

  get_by_id = async (id: string) => {
    // console.log(payload);
    return axios.get(`${ip}/${this.url}`, { params: { answer_id: id } });
  };

  add = async (payload: T) => {
    return axios.post(`${ip}/${this.url}`, payload);
  };

  addWithImage = async (payload: T) => {
    return axios.post(`${ip}/files/images`, payload);
  };

  addFile = async (payload: T) => {
    return axios.post(`${ip}/files/answer`, payload);
  };

  add2 = async (payload: T) => {
    Object.keys(payload).forEach((key) => {
      // payload[key] = payload[key];
      payload[key] = payload[key];
    });
    return axios.post(`${ip}/${this.url}`, payload);
  };

  upd = async (payload: T & { id: string | undefined, data: FormData }) => {
    const { id } = payload;
    payload.id = undefined;
    return axios.put(`${ip}/update_sort_answer_text`, payload.data);
  };

  updWithImage = async (payload: T & { id: string | undefined, data: FormData }) => {
    const { id } = payload;
    payload.id = undefined;
    return axios.put(`${ip}/update_sort_answer_image`, payload.data);
  };

}

const answer = new Answer({ name: 'Answer', url: 'answers' });

export default answer;
